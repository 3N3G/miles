"""Measure agreement between the two saved judge orders after group centering.

This describes hypothetical single-order reward contrasts before token-level
whitening. Training used the average of both orders. Neither this contrast nor
its cosine measures mathematical correctness or actual gradient agreement.
"""

import argparse
import collections
import datetime
import hashlib
import json
import math
import statistics
from pathlib import Path

from experiments.tcs_async_rl.diagnose_rl import save


def summarize(samples):
    grouped = collections.defaultdict(list)
    identities = set()
    for row in samples:
        identity = (row["optimizer_step"], row["sample_index"])
        if identity in identities:
            raise ValueError("Duplicate consumed answer")
        identities.add(identity)
        a, b = row["order0"], row["order1"]
        if not all(math.isfinite(value) for value in (a, b, row["reward"])):
            raise ValueError("Nonfinite judge score")
        if (a + b) / 2 != row["reward"]:
            raise ValueError("Consumed reward does not average the two judge orders")
        grouped[(row["optimizer_step"], row["question_key"])].append(row)

    groups, answers = [], []
    for (step, question), rows in sorted(grouped.items()):
        means = [statistics.mean(row[f"order{i}"] for row in rows) for i in (0, 1)]
        centered = [[row[f"order{i}"] - means[i] for row in rows] for i in (0, 1)]
        norms = [math.fsum(value * value for value in order) for order in centered]
        cosine = (
            math.fsum(a * b for a, b in zip(*centered, strict=True)) / math.sqrt(norms[0] * norms[1])
            if all(norms)
            else None
        )
        current = []
        for row, a, b in zip(rows, *centered, strict=True):
            result = {
                "optimizer_step": step,
                "question_key": question,
                "sample_index": row["sample_index"],
                "order_scores": [row["order0"], row["order1"]],
                "centered_scores": [a, b],
                "strict_sign_reversal": a * b < 0,
                "one_order_tie": (a == 0) != (b == 0),
                "both_orders_tie": a == b == 0,
            }
            current.append(result)
        answers.extend(current)
        groups.append(
            {
                "optimizer_step": step,
                "question_key": question,
                "answers": len(rows),
                "mean_by_order": means,
                "constant_orders": sum(norm == 0 for norm in norms),
                "centered_score_cosine": cosine,
                "strict_sign_reversals": sum(row["strict_sign_reversal"] for row in current),
                "one_order_ties": sum(row["one_order_tie"] for row in current),
            }
        )
    cosines = [row["centered_score_cosine"] for row in groups if row["centered_score_cosine"] is not None]
    return {
        "answers": len(samples),
        "groups": len(groups),
        "score_disagreements": sum(row["order0"] != row["order1"] for row in samples),
        "strict_sign_reversals": sum(row["strict_sign_reversal"] for row in answers),
        "one_order_ties": sum(row["one_order_tie"] for row in answers),
        "both_orders_tie": sum(row["both_orders_tie"] for row in answers),
        "groups_with_defined_cosine": len(cosines),
        "mean_group_cosine": statistics.mean(cosines) if cosines else None,
        "minimum_group_cosine": min(cosines, default=None),
        "per_group": groups,
        "per_answer": answers,
        "limits": [
            "Both orders were averaged for actual training. These are hypothetical single-order centered contrasts.",
            "Token whitening, response lengths, importance weights and gradients are not reconstructed here.",
            "Presentation order and stochastic judge variation are confounded without same-order repeated calls.",
            "Agreement does not establish mathematical correctness. Groups and answers are not independent seeds.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    if cli.output.exists():
        raise FileExistsError("Preserve the old diagnostic and use a new output path")
    raw = cli.samples.read_bytes()
    result = summarize(json.loads(raw))
    result.update(
        at_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        source_samples=str(cli.samples),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        new_provider_calls=0,
    )
    save(cli.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("per_group", "per_answer")}))


if __name__ == "__main__":
    main()
