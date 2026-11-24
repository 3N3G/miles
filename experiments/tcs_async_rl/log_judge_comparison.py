"""Publish an already completed judge comparison, with numeric readback verification."""

import argparse
import json
import math
import time
from pathlib import Path

from experiments.tcs_async_rl.rl_artifacts import _atomic_bytes


def verify(directory, path, metrics, packet_sha256):
    import wandb

    for attempt in range(30):
        remote = wandb.Api().run(path)
        summary = dict(remote.summary)
        if remote.state == "finished" and all(
            key in summary and math.isclose(summary[key], value, rel_tol=0, abs_tol=1e-6)
            for key, value in metrics.items()
        ):
            output = {
                "url": remote.url,
                "state": remote.state,
                "verified_scalars": len(metrics),
                "packet_sha256": packet_sha256,
                "optimizer_updates": 0,
            }
            _atomic_bytes(directory / "wandb-judge-comparison.json", (json.dumps(output, indent=2) + "\n").encode())
            return output
        if attempt < 29:
            time.sleep(3)
    raise RuntimeError("W&B comparison metrics failed readback; retain the run and retry verification")


def publish(directory, session_uuid):
    # Authentication remains inside the already configured local SDK.
    import wandb

    receipt = directory / "wandb-judge-comparison.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    high = json.loads((directory / "judgments-high-v1/comparison.json").read_text())
    medium = json.loads((directory / "judgments-v2/comparison.json").read_text())
    controls = json.loads((directory / "judgments-high-v1/controls.json").read_text())
    freeze = json.loads((directory / "judgments-high-v1/pre-call-label-freeze.json").read_text())
    if controls["failures"] or not high["reference_guided"]["passes_observed_gate"]:
        raise ValueError("Only the completed passing comparison is supported by this logger")
    methods = {
        "anonymous_medium": high["anonymous_pair"],
        "reference_medium": medium["reference_guided"],
        "reference_high": high["reference_guided"],
    }
    if any(
        [row["case"] for row in data["cases"]] != [row["case"] for row in methods["reference_high"]["cases"]]
        for data in methods.values()
    ):
        raise ValueError("Compared methods use different cases")
    metrics = {
        f"judge/{name}/{metric}": data[metric]
        for name, data in methods.items()
        for metric in ("exact_score_agreement", "case_balanced_item_agreement", "mean_absolute_score_error")
    }
    metrics.update(
        {
            "controls/passed": controls["presentations"],
            "controls/failed": 0,
            "judge/always_full_credit_exact_agreement": sum(
                row["audit_score"] == 7 for row in methods["reference_high"]["cases"]
            )
            / 40,
            "comparison/optimizer_updates": 0,
        }
    )
    config = {
        "stage": "fixed_saved_answer_judge_validation",
        "judge_model": "gpt-6-luna",
        "selected_effort": "high",
        "selected_grading_method": "reference_guided",
        "answers": 40,
        "question_clusters": 17,
        "packet_sha256": freeze["packet_sha256"],
        "labels_sha256": freeze["labels_sha256"],
        "selected_prompt_sha256": freeze["prompt_sha256"]["rubric_judge_reference_v2.txt"],
        "auditor": "Assistant, not independent human calibration",
        "all_eight_medium_audits_score_zero": True,
        "exact_agreement_gate": 0.9,
        "item_agreement_gate": 0.95,
        "approved_all_stages_judge_usd_cap": 25,
    }
    client = directory.parent / ".wandb-clients" / "judge-high-20261124"
    client.mkdir(parents=True, exist_ok=True)
    owner_file = directory / "wandb-judge-started.json"
    if owner_file.exists():
        owner = json.loads(owner_file.read_text())
        if owner["session_uuid"] != session_uuid or owner["packet_sha256"] != freeze["packet_sha256"]:
            raise ValueError("The existing comparison run belongs to a different session or packet")
        existing = wandb.Api().run("geney/tcs-async-rl/judge-high-20261124")
        if any(key in dict(existing.summary) for key in metrics):
            return verify(directory, "geney/tcs-async-rl/judge-high-20261124", metrics, freeze["packet_sha256"])
    figure = directory.parent / "judge-high-figures-20261124/judge-effort-comparison.png"
    image = wandb.Image(str(figure))  # Validate image dependencies before creating a remote run.
    run = wandb.init(
        entity="geney",
        project="tcs-async-rl",
        id="judge-high-20261124",
        name="Luna high: fixed-answer validation",
        group="judge-comparison-20261124",
        job_type="judge-validation",
        config=config,
        dir=str(client),
        resume="must" if owner_file.exists() else "never",
        settings=wandb.Settings(mode="online", x_update_finish_state=True),
    )
    path = "/".join((run.entity, run.project, run.id))
    _atomic_bytes(
        owner_file,
        (
            json.dumps(
                {"session_uuid": session_uuid, "packet_sha256": freeze["packet_sha256"], "url": run.url}, indent=2
            )
            + "\n"
        ).encode(),
    )
    table = wandb.Table(columns=["case", "assistant_audit", *methods])
    for index, row in enumerate(methods["reference_high"]["cases"]):
        table.add_data(row["case"], row["audit_score"], *[data["cases"][index]["reward"] for data in methods.values()])
    run.log({**metrics, "judge/comparison_table": table, "judge/comparison_figure": image})
    run.finish()
    return verify(directory, path, metrics, freeze["packet_sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--session-uuid", required=True)
    cli = parser.parse_args()
    print(json.dumps(publish(cli.directory, cli.session_uuid), indent=2))
