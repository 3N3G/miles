"""Predeclared paired accuracy analysis and control gate. No new sampling."""

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from experiments.tcs_async_rl.gsm8k_artifacts import write_json
from experiments.tcs_async_rl.native_metrics import read_native_metrics


def paired_accuracy(before, after, *, seed=20261126, bootstrap_samples=10000):
    if not before or before.keys() != after.keys():
        raise ValueError("Paired accuracy requires identical nonempty question IDs")
    if any(x not in (0, 1) for scores in (before, after) for x in scores.values()):
        raise ValueError("Exact accuracy requires binary scores")
    keys = sorted(before)
    gains = [after[key] - before[key] for key in keys]
    improved, regressed = gains.count(1), gains.count(-1)
    discordant = improved + regressed
    probability = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1))
                      / 2**discordant) if discordant else 1.0
    rng = random.Random(seed)
    boot = sorted(statistics.mean(rng.choices(gains, k=len(gains))) for _ in range(bootstrap_samples))
    return {
        "questions": len(keys), "before_accuracy": statistics.mean(before.values()),
        "after_accuracy": statistics.mean(after.values()), "change": statistics.mean(gains),
        "wrong_to_right": improved, "right_to_wrong": regressed,
        "mcnemar_two_sided_exact_p": probability, "bonferroni_two_comparisons_p": min(1, 2 * probability),
        "paired_problem_bootstrap_95_interval": [boot[int(.025 * len(boot))], boot[int(.975 * len(boot))]],
        "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
        "uncertainty_unit": "Validation questions. One training seed. Not training-seed uncertainty.",
    }


def analyze(directory):
    complete = json.loads((directory / "control-complete.json").read_text())
    if complete["uninformative"]:
        result = {"gate_passed": False, "uninformative": True,
                  "details": json.loads((directory / "uninformative.json").read_text())}
    else:
        before = json.loads((directory / "evaluations/baseline.json").read_text())
        updates = complete["optimizer_updates"]
        after = json.loads((directory / f"evaluations/updates-{updates:06d}.json").read_text())
        # A deadline save can leave the next prefetched rollout untrained.
        # Require the complete published optimizer prefix, then match its count
        # to the driver's final checkpoint/evaluation boundary below.
        native = read_native_metrics(directory, allow_pending=True, require_publication=True)
        if len(native) != updates or [x["optimizer_step"] for x in native] != list(range(updates)):
            raise ValueError("Final evaluation and actual optimizer history differ")
        direction = [row["train/update_direction/surrogate_gain"] for row in native
                     if "train/update_direction/surrogate_gain" in row]
        if not direction or not all(math.isfinite(value) for value in direction):
            raise ValueError("Missing finite local update-direction evidence")
        paired = paired_accuracy(before["scores"], after["scores"])
        result = {"optimizer_updates": updates, "paired": paired,
                  "direction_checks": direction, "mean_direction_gain": statistics.mean(direction),
                  "gate_passed": paired["change"] > 0 and statistics.mean(direction) > 0,
                  "baseline": {key:value for key,value in before.items() if key != "scores"},
                  "final": {key:value for key,value in after.items() if key != "scores"}}
    write_json(directory / "control-results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    print(json.dumps(analyze(parser.parse_args().directory), indent=2))
