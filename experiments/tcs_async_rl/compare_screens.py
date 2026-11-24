"""Compare fixed-question checkpoint screens under the same grading recipe.

These are descriptive observations from one sampling run per checkpoint. The
questions can enter training. Neither this table nor its plots measure held-out
generalization or seed-level uncertainty.
"""

import argparse
import csv
import json
import math
import shutil
import statistics
from pathlib import Path

from experiments.tcs_async_rl.rl_artifacts import file_sha256


def validate_summary(summary):
    rows = summary["questions"]
    if (
        summary["optimizer_updates"] != 0
        or summary["grade_method"] != "reference_guided"
        or summary["judge_reasoning_effort"] != "high"
        or summary["answers"] != 384
        or len(rows) != 12
        or len({row["question_key"] for row in rows}) != 12
        or sorted(row["difficulty"] for row in rows) != ["easy"] * 6 + ["medium"] * 6
    ):
        raise ValueError("Expected a complete fixed twelve-question Luna-high screen")
    for row in rows:
        counts = {float(score): count for score, count in row["reward_counts"].items()}
        if (
            row["answers"] != 32
            or not counts
            or len(counts) != len(row["reward_counts"])
            or any(not math.isfinite(score) or not 0 <= score <= 7 for score in counts)
            or any(type(count) is not int or count <= 0 for count in counts.values())
            or sum(counts.values()) != 32
            or counts.get(7, 0) != row["full_credit"]
            or not math.isclose(sum(score * count for score, count in counts.items()) / 32, row["mean_reward"])
            or type(row["capped"]) is not int
            or not 0 <= row["capped"] <= 32
            or not math.isfinite(row["mean_response_tokens"])
            or row["mean_response_tokens"] < 0
        ):
            raise ValueError("Screen summary differs from its per-question score counts")


def compare(before, after=None):
    validate_summary(before)
    if after is not None:
        validate_summary(after)
        for key in ("dataset_sha256", "judge_prompt_sha256", "judge_config_sha256", "sampling"):
            if before[key] != after[key]:
                raise ValueError("Screen comparison changed its questions or grading recipe")
        if after["checkpoint_rollout_id"] <= before["checkpoint_rollout_id"]:
            raise ValueError("The after screen must use a later saved checkpoint")
        if {row["question_key"] for row in before["questions"]} != {row["question_key"] for row in after["questions"]}:
            raise ValueError("Screen comparison has different question identities")
    later = {row["question_key"]: row for row in after["questions"]} if after else {}
    rows, indices = [], {"easy": 0, "medium": 0}
    for first in before["questions"]:
        last = later.get(first["question_key"])
        difficulty = first["difficulty"]
        indices[difficulty] += 1
        if last and any(first[key] != last[key] for key in ("paper_id", "difficulty")):
            raise ValueError("Question metadata changed between screens")
        row = {
            "label": f"{difficulty[0].upper()}{indices[difficulty]}",
            "question_key": first["question_key"],
            "paper_id": first["paper_id"],
            "difficulty": difficulty,
        }
        for key in ("mean_reward", "full_credit", "capped", "mean_response_tokens"):
            row[f"before_{key}"] = first[key]
            row[f"after_{key}"] = last[key] if last else None
            row[f"change_{key}"] = last[key] - first[key] if last else None
        rows.append(row)
    aggregates = {}
    for difficulty in indices:
        members = [row for row in rows if row["difficulty"] == difficulty]
        aggregates[difficulty] = {
            "questions": len(members),
            "answers_per_checkpoint": 32 * len(members),
            **{
                f"{phase}_mean_reward": (
                    statistics.mean(row[f"{phase}_mean_reward"] for row in members)
                    if phase == "before" or after
                    else None
                )
                for phase in ("before", "after", "change")
            },
        }
    return {"questions": rows, "by_difficulty": aggregates, "paired": after is not None}


def plot_comparison(result, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = result["questions"]
    for key, axis, maximum in (
        ("mean_reward", "Mean rubric score (points)", 7.15),
        ("full_credit", "Full-credit answers out of 32", 33),
        ("capped", "Answers reaching the 90,000-token limit, out of 32", 33),
        ("mean_response_tokens", "Mean generated response length (tokens)", None),
    ):
        fig, ax = plt.subplots(figsize=(9, 7))
        positions = list(range(len(rows)))
        ax.scatter([row[f"before_{key}"] for row in rows], positions, color="#666666", label="Before training")
        if result["paired"]:
            for index, row in enumerate(rows):
                ax.plot([row[f"before_{key}"], row[f"after_{key}"]], [index, index], color="#BBBBBB", zorder=0)
            ax.scatter(
                [row[f"after_{key}"] for row in rows], positions, color="#0072B2", marker="s", label="After training"
            )
        ax.set(
            xlabel=axis,
            yticks=positions,
            yticklabels=[f"{row['label']} ({row['difficulty']})" for row in rows],
            xlim=(0, maximum),
        )
        ax.set_title("Fixed-question checkpoint screen under Luna high", pad=40)
        ax.invert_yaxis()
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False)
        fig.text(
            0.5,
            0.01,
            "32 answers per question. One sampling run per checkpoint. No inferential error bars.",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(output / f"screen-{key}.png", dpi=180, bbox_inches="tight")
        fig.savefig(output / f"screen-{key}.svg", bbox_inches="tight")
        plt.close(fig)


def write_report(result, output):
    lines = [
        "# Fixed-question checkpoint screen",
        "",
        "This screen asks whether the saved Qwen3.5-4B policy changes its scores on the same twelve questions.",
        "Each checkpoint generates 32 answers per question, on six easy and six medium questions.",
        "GPT-6 Luna at high effort grades the final answer against the supplied reference and weighted binary rubric.",
        "Scores range from zero to seven points. The input, grading recipe and sampling settings must match.",
        "",
        "| Difficulty | Questions | Answers per checkpoint | Before mean | After mean | Change |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for difficulty, values in result["by_difficulty"].items():
        scores = [
            f"{values[f'{phase}_mean_reward']:.3f}" if values[f"{phase}_mean_reward"] is not None else "Unrun"
            for phase in ("before", "after", "change")
        ]
        lines.append(
            f"| {difficulty} | {values['questions']} | {values['answers_per_checkpoint']} | {' | '.join(scores)} |"
        )
    lines.extend(
        [
            "",
            "One sampling run per checkpoint does not measure uncertainty across training seeds.",
            "The plots are descriptive and have no inferential error bars.",
            "Easy questions were previously trained. Medium questions can enter this continuation.",
            "These scores do not measure held-out generalization. FOCS 2026 and STOC 2026 remain sealed.",
            "The judge was checked against assistant audits, without independent human calibration.",
            "",
            "Every question has one uniformly selected raw answer per checkpoint below.",
            "An example is an individual answer, not its question's mean score.",
            "The labels E and M identify easy and medium questions. Full paper names are in the CSV.",
            "",
            "| Question | Before mean | Before raw answer | After mean | After raw answer |",
            "|---|---:|---|---:|---|",
        ]
    )
    for row in result["questions"]:
        after = f"{row['after_mean_reward']:.3f}" if result["paired"] else "Unrun"
        raw = f"[Answer](examples/{row['label']}-after.json)" if result["paired"] else "Unrun"
        lines.append(
            f"| {row['label']} | {row['before_mean_reward']:.3f} | [Answer](examples/{row['label']}-before.json) | {after} | {raw} |"
        )
    lines.extend(["", "[Per-question numeric values](per-question.csv). [Source fingerprints](comparison.json).", ""])
    (output / "README.md").write_text("\n".join(lines))


def write_comparison(before_directory, after_directory, output):
    summaries = [json.loads((before_directory / "screen-summary.json").read_text())]
    if after_directory:
        summaries.append(json.loads((after_directory / "screen-summary.json").read_text()))
    result = compare(*summaries)
    output.mkdir(parents=True, exist_ok=True)
    (output / "examples").mkdir(exist_ok=True)
    provenance = []
    for phase, directory, summary in zip(
        ("before", "after"), (before_directory, after_directory), summaries, strict=False
    ):
        provenance.append(
            {
                "phase": phase,
                "checkpoint_rollout_id": summary["checkpoint_rollout_id"],
                "summary_sha256": file_sha256(directory / "screen-summary.json"),
                "responses_sha256": summary["responses_sha256"],
            }
        )
        examples = {row["question_key"]: row["seeded_example"] for row in summary["questions"]}
        for row in result["questions"]:
            source = directory / examples[row["question_key"]]
            if source.resolve().parent != (directory / "seeded-examples").resolve():
                raise ValueError("Raw example escaped the screen example directory")
            shutil.copyfile(source, output / "examples" / f"{row['label']}-{phase}.json")
    result.update(
        provenance=provenance,
        dataset_sha256=summaries[0]["dataset_sha256"],
        judge_prompt_sha256=summaries[0]["judge_prompt_sha256"],
    )
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "per-question.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["questions"][0]))
        writer.writeheader()
        writer.writerows(result["questions"])
    plot_comparison(result, output)
    write_report(result, output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    print(json.dumps(write_comparison(cli.before, cli.after, cli.output), indent=2))
