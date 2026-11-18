"""Write descriptive group summaries, plots, and a reproducible raw example."""

import argparse
import json
import random
from pathlib import Path


def analyze(run_dir):
    responses = [json.loads(line) for line in (run_dir / "responses.jsonl").read_text().splitlines()]
    grades_path = run_dir / "group-grades.json"
    grades = json.loads(grades_path.read_text()) if grades_path.exists() else None
    selected = random.Random(20261118).choice(sorted(responses, key=lambda row: row["sample_index"]))
    (run_dir / "raw-example.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")
    if grades is not None:
        cases = grades["scores_by_case"]
        packet = random.Random(20261118).sample(responses, min(10, len(responses)))
        (run_dir / "human-calibration-packet.json").write_text(
            json.dumps(
                [
                    {
                        "sample_index": row["sample_index"],
                        "answer": row["final_answer"],
                        "human_score": None,
                        "notes": "",
                    }
                    for row in packet
                ],
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        scores = [value for key, value in cases.items() if key.startswith("sample-")]
    else:
        scores = []
    lengths = [row["output"]["meta_info"]["completion_tokens"] for row in responses]
    summary = {
        "completed": len(responses),
        "with_final_answer": sum(bool(row["final_answer"]) for row in responses),
        "lengths": lengths,
        "provisional_scores": scores,
        "raw_example_sample_index": selected["sample_index"],
        "limits": "One question and one seed. Provisional LLM grades, without human calibration or independent provider verification. No population success rate or learning effect.",
    }
    (run_dir / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Plot dependencies are optional for operational import and CPU unit tests.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for values, filename, label, bins in (
        (lengths, "response-lengths.png", "Generated tokens per response", 12),
        (scores, "provisional-grades.png", "Provisional rubric points (0–7)", [i / 2 - 0.25 for i in range(16)]),
    ):
        if not values:
            continue
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.hist(values, bins=bins, color="#0072B2", edgecolor="white")
        axis.set_xlabel(label)
        axis.set_ylabel("Responses (count)")
        axis.set_ylim(bottom=0)
        figure.tight_layout()
        figure.savefig(run_dir / filename, dpi=180)
        plt.close(figure)
    row = json.loads((run_dir / "input.json").read_text())
    report = [
        "# First TCS inference group",
        "",
        "Question: does frozen Qwen3.5-4B produce complete answers and varied rubric grades on one eligible question?",
        "",
        "Setup: 32 responses, thinking enabled, 90,000-token output cap, temperature one, no learning. The single eligible question was selected with seed 20261118. FOCS/STOC 2026 remain untouched.",
        "",
        f"Completed responses: {summary['completed']}/32. Responses with final answers: {summary['with_final_answer']}. These counts concern one question, not independent papers.",
        "",
        "Provisional Luna medium scores, averaged over answer order: " + json.dumps(scores) + ".",
        "",
        summary["limits"],
        "",
        "![Generated lengths](response-lengths.png)",
        "",
        "![Provisional grades](provisional-grades.png)",
        "",
        "## Raw problem",
        "",
        row["problem"],
        "",
        f"## Seed-selected raw response {selected['sample_index']}",
        "",
        selected["output"].get("text", ""),
        "",
        "The full input, reference, rubric, per-item decisions, both presentation orders, token counts, and receipts are included in this private run directory. No interpretation or correctness claim follows from judge agreement.",
        "",
    ]
    (run_dir / "REPORT.md").write_text("\n".join(report))
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    analyze(parser.parse_args().run_dir)


if __name__ == "__main__":
    main()
