"""Organize an existing W&B workspace without writing run history.

Run with wandb==0.31.0 and wandb-workspaces==0.4.13. The default writes a
reviewable backup and proposal. Pass --apply against that same directory to
save the proposal after checking for concurrent workspace edits.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import wandb
from wandb_workspaces import workspaces as ws
from wandb_workspaces._graphql import execute_graphql
from wandb_workspaces.reports import v2 as panels

AXIS = "audit/optimizer_step"
EVAL_AXIS = "control/eval_optimizer_updates"
GSM8K_SECTION = "GSM8K control"
VIEW_QUERY = """
query DashboardViews($entity: String!, $project: String!) {
  project(name: $project, entityName: $entity) {
    allViews(viewType: "project-view") {
      edges { node { id name displayName type spec } }
    }
  }
}
"""


def read_view(api, entity, project, name):
    result = execute_graphql(api, VIEW_QUERY, {"entity": entity, "project": project})
    matches = [edge["node"] for edge in result["project"]["allViews"]["edges"] if edge["node"]["name"] == name]
    if len(matches) != 1:
        raise ValueError("Expected exactly one existing workspace with the requested name")
    return matches[0]


def line(title, metrics, unit, limits=(None, None)):
    return panels.LinePlot(
        title=title,
        x=AXIS,
        y=metrics,
        title_x="Optimizer update",
        title_y=unit,
        range_y=limits,
        smoothing_type="none",
        smoothing_factor=0,
        aggregate=False,
    )


def key_sections(*, difficulty_breakdown=False):
    definitions = [
        (
            "Return",
            [
                line(
                    "Mean return — raw rubric reward",
                    ["audit/rollout/episode_raw_reward"],
                    "Points (0–7)",
                    (0, 7),
                )
            ],
            1,
        ),
        (
            "Policy health",
            [
                line(
                    "Full and sampling entropy",
                    ["audit/train/full_policy_entropy", "audit/train/sampling_entropy"],
                    "Nats per token",
                ),
                line("Divergence from initialization", ["audit/train/kl_loss"], "Sampled k3 estimate"),
                line("Trainer–rollout mismatch", ["audit/train/train_rollout_kl"], "Sampled k3 estimate"),
                line("Gradient norm", ["audit/train/grad_norm"], "Norm"),
            ],
            2,
        ),
        (
            "Generation",
            [
                line("Mean response length", ["audit/rollout/response_lengths"], "Tokens"),
                line("Answers reaching the token cap", ["audit/rollout/truncated_ratio"], "Fraction", (0, 1)),
            ],
            2,
        ),
        (
            "Run status",
            [
                panels.ScalarChart(title="Total saved updates", metric="monitor/saved_total_optimizer_updates"),
                panels.ScalarChart(title="New saved updates", metric="monitor/new_durable_updates"),
                panels.ScalarChart(title="Shared judge spend (USD)", metric="monitor/shared_judge_settled_usd"),
            ],
            3,
        ),
    ]
    if difficulty_breakdown:
        definitions[0][1].append(
            line(
                "Raw return by question difficulty",
                [f"audit/rollout/by_difficulty/{level}/raw_reward" for level in ("easy", "medium")],
                "Points (0–7)",
                (0, 7),
            )
        )
        definitions.append(
            (
                "Training coverage",
                [
                    line(
                        "Fraction of questions consumed",
                        ["audit/rollout/corpus_coverage_fraction"],
                        "Fraction",
                        (0, 1),
                    ),
                    line(
                        "Groups with constant reward",
                        ["audit/rollout/constant_reward_group_fraction"],
                        "Fraction",
                        (0, 1),
                    ),
                ],
                2,
            )
        )
    return [
        ws.Section(
            name=name,
            panels=charts,
            is_open=True,
            pinned=name == "Return",
            layout_settings=ws.SectionLayoutSettings(columns=columns, rows=(2 if difficulty_breakdown else 1) if name == "Return" else 2),
            panel_settings=ws.SectionPanelSettings(x_axis=AXIS),
        )
        ._to_model()
        .model_dump(mode="json", by_alias=True, exclude_none=True)
        for name, charts, columns in definitions
    ]


def gsm8k_sections():
    definitions = [
        ("Fixed-validation exact accuracy", EVAL_AXIS, ["control/eval/accuracy"]),
        ("Consumed training reward", AXIS, ["audit/rollout/episode_raw_reward"]),
        ("Constant-reward training groups", AXIS, ["audit/rollout/zero_std/all_one_percentage", "audit/rollout/zero_std/all_zero_percentage"]),
        ("Validation answers reaching the token cap", EVAL_AXIS, ["control/eval/capped_fraction"]),
    ]
    charts = [
        panels.LinePlot(
            title=title,
            x=axis,
            y=metrics,
            title_x="Completed optimizer updates" if axis == EVAL_AXIS else "Optimizer update (zero-based)",
            title_y="Fraction",
            range_y=(0, 1),
            smoothing_type="none",
            smoothing_factor=0,
            aggregate=False,
        )
        for title, axis, metrics in definitions
    ]
    return [
        ws.Section(
            name=GSM8K_SECTION,
            panels=charts,
            is_open=True,
            pinned=True,
            layout_settings=ws.SectionLayoutSettings(columns=2, rows=2),
        )
        ._to_model()
        .model_dump(mode="json", by_alias=True, exclude_none=True)
    ]


def updated_spec(original, *, difficulty_breakdown=False, gsm8k_control=False):
    spec = copy.deepcopy(original)
    old = spec["section"]["panelBankConfig"]["sections"]
    by_name = {section["name"]: section for section in old}
    selected = gsm8k_sections() if gsm8k_control else key_sections(difficulty_breakdown=difficulty_breakdown)
    for section in selected:
        previous = by_name.get(section["name"], by_name.get("monitor", {}) if section["name"] == "Run status" else {})
        if "__id__" in previous:
            section["__id__"] = previous["__id__"]
        section["isPanelsAuto"] = False
        for panel in section["panels"]:
            panel["isAuto"] = False
        if section["name"] == "Run status":
            section["defaultName"] = "monitor"
    managed = {section["name"] for section in selected} | (set() if gsm8k_control else {"monitor"})
    remaining = [section for section in old if section["name"] not in managed]
    for section in remaining:
        if gsm8k_control:
            continue  # Preserve all existing TCS panels and their open/closed state.
        section["isOpen"] = False
        if section["name"] == "audit":
            section.setdefault("localPanelSettings", {}).update(xAxis=AXIS, xAxisActive=True)
    spec["section"]["panelBankConfig"]["sections"] = selected + remaining
    return spec


def validate_panels(spec, history, summary, *, gsm8k_control=False):
    counts = {}
    for section in spec["section"]["panelBankConfig"]["sections"]:
        if gsm8k_control and section["name"] != GSM8K_SECTION:
            continue
        if section.get("isPanelsAuto") is not False or section["name"] == "Hidden Panels":
            continue
        for panel in section["panels"]:
            config = panel["config"]
            for metric in config["metrics"]:
                if panel["viewType"] == "Scalar Chart":
                    if not isinstance(summary.get(metric), (int, float)):
                        raise ValueError(f"Summary card has no numeric value: {metric}")
                    counts[metric] = summary[metric]
                else:
                    expected_axis = EVAL_AXIS if metric.startswith("control/eval/") else AXIS
                    if config["xAxis"] != expected_axis:
                        raise ValueError("A key chart has the wrong optimizer axis")
                    points = [row for row in history if expected_axis in row and metric in row]
                    if not points or not all(math.isfinite(row[metric]) for row in points):
                        raise ValueError(f"Chart has no finite aligned values: {metric}")
                    counts[metric] = len(points)
    return counts


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--view-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--difficulty-breakdown", action="store_true")
    parser.add_argument("--gsm8k-control", action="store_true")
    cli = parser.parse_args()
    cli.output.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    view = read_view(api, cli.entity, cli.project, cli.view_name)
    before = cli.output / "view-before.json"
    proposed = cli.output / "spec-proposed.json"
    if before.exists():
        if json.loads(before.read_text()) != view:
            raise ValueError("Workspace changed since the saved proposal; prepare a fresh proposal")
    else:
        save_json(before, view)
    if not proposed.exists():
        save_json(proposed, updated_spec(json.loads(view["spec"]), difficulty_breakdown=cli.difficulty_breakdown, gsm8k_control=cli.gsm8k_control))
    spec = json.loads(proposed.read_text())
    run_path = f"{cli.entity}/{cli.project}/{cli.run_id}"
    run = api.run(run_path)
    history = list(run.scan_history(use_cache=False))
    digest = hashlib.sha256(json.dumps(history, sort_keys=True).encode()).hexdigest()
    counts = validate_panels(spec, history, dict(run.summary), gsm8k_control=cli.gsm8k_control)
    if cli.apply:
        query = """
        mutation SaveDashboard($id: ID!, $entity: String!, $project: String!, $name: String!, $display: String!, $spec: String!) {
          upsertView(input: {id: $id, entityName: $entity, projectName: $project,
            name: $name, displayName: $display, type: "project-view", spec: $spec, createdUsing: WANDB_SDK}) {
            view { id name } inserted
          }
        }
        """
        response = execute_graphql(
            api,
            query,
            {
                "id": view["id"],
                "entity": cli.entity,
                "project": cli.project,
                "name": view["name"],
                "display": view["displayName"],
                "spec": json.dumps(spec),
            },
        )
        if response["upsertView"]["inserted"] or response["upsertView"]["view"]["id"] != view["id"]:
            raise RuntimeError("Dashboard update did not preserve the existing workspace")
        after = read_view(api, cli.entity, cli.project, cli.view_name)
        save_json(cli.output / "view-after.json", after)
        if json.loads(after["spec"]) != spec:
            raise RuntimeError("Saved dashboard differs from its proposal")
        remote = api.run(run_path)
        actual = list(remote.scan_history(use_cache=False))
        if hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest() != digest or remote.state != run.state:
            raise RuntimeError("Run history or terminal state changed during a dashboard-only update")
    receipt = {
        "applied": cli.apply,
        "view_id": view["id"],
        "run_path": run_path,
        "history_rows": len(history),
        "history_sha256": digest,
        "validated_metrics": counts,
        "sections": [section["name"] for section in spec["section"]["panelBankConfig"]["sections"]],
    }
    save_json(cli.output / ("verification.json" if cli.apply else "proposal-verification.json"), receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
