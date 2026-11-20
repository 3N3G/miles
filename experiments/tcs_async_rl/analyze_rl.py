"""Descriptive figures for the actual trainer-consumed groups, without learning claims."""

import argparse
import json
from pathlib import Path


def analyze(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = json.loads((directory / "runtime-acceptance.json").read_text())
    groups = result.get("consumed_groups", [])
    paragraphs = [
        "# Two-update asynchronous RL runtime check\n",
        "This check asks whether the AMD deployment can complete two optimizer updates with real rubric rewards.",
        "The setup uses Qwen3.5-4B with thinking enabled, a 90,000-token output limit, and one preselected medium TCS question.",
        "CISPO trains on groups of 32 answers. Two MI350 GPUs train while six generate answers.",
        "GPT-6 Luna at medium effort grades final answers against the supplied seven-point rubric in both anonymous orders.\n",
        f"Automated runtime acceptance passed: **{result['runtime_acceptance_passed']}**.",
        f"The saved evidence contains {len(result.get('training', {}).get('valid_updates', []))} valid optimizer-step records",
        f"and {len(groups)} exported trainer-consumed groups. See [all checks](runtime-acceptance.json).\n",
    ]
    for index, group in enumerate(groups):
        scores = sorted((float(score), count) for score, count in group["reward_histogram"].items())
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([x[0] for x in scores], [x[1] for x in scores], width=0.4, color="#0072B2")
        ax.set(
            xlabel="Provisional rubric points (0–7)",
            ylabel="Responses",
            xlim=(-0.5, 7.5),
            title=f"Consumed group {index}: observed reward distribution",
        )
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(directory / f"consumed-{index}-rewards.png", dpi=180)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(group["response_lengths"], bins=8, color="#0072B2")
        ax.axvline(90000, color="#D55E00", linestyle="--", label="Output limit")
        ax.set(xlabel="Generated tokens", ylabel="Responses", title=f"Consumed group {index}: generation cost")
        ax.set_ylim(bottom=0)
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / f"consumed-{index}-lengths.png", dpi=180)
        plt.close(fig)
        paragraphs.extend(
            [
                f"Group {index} consumed {group['samples']} answers. Score counts were {dict(scores)}.",
                f"[Reward plot](consumed-{index}-rewards.png), [token-count plot](consumed-{index}-lengths.png),",
                f"and [seed-selected raw answer with its reward]({group['example_file']}).\n",
            ]
        )
    paragraphs.extend(
        [
            "These are runtime checks on one question, with one training seed. They do not estimate learning gains or corpus performance.",
            "The plots describe individual responses, with no error bars or significance test.",
            "Human judge calibration and an independent-provider audit remain pending. Grade differences can include judge error.",
            "Inspect actual per-token weight versions before claiming generation used updated weights.",
            "The next step is to review failed checks or inspect the completed runtime evidence before selecting a corpus training plan.",
        ]
    )
    (directory / "REPORT.md").write_text("\n".join(paragraphs) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    analyze(parser.parse_args().directory)


if __name__ == "__main__":
    main()
