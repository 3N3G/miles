"""Independently check numeric equality after the native box parser. Never change rewards."""

import argparse
import collections
import gzip
import json
import re
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from experiments.tcs_async_rl.gsm8k_artifacts import write_json


def rational(expression):
    if not isinstance(expression, str):
        return None
    text = expression.strip().replace(",", "")
    if re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)", text):
        return Fraction(Decimal(text))
    match = re.fullmatch(r"\\frac\{([+-]?[0-9]+)\}\{([+-]?[0-9]+)\}", text)
    if match and int(match[2]) != 0:
        return Fraction(int(match[1]), int(match[2]))
    return None


def check_value(answer, label):
    if answer is None:
        return 0, "missing_parsed_box"
    target, actual = rational(str(label)), rational(answer)
    if target is None:
        return None, "unsupported_label"
    if actual is not None:
        method = "fraction" if "\\frac" in answer else "decimal"
        return int(target == actual), method
    match = re.fullmatch(r"([0-9]*)\\sqrt\{([0-9]+)\}", answer)
    if match:
        coefficient = int(match[1]) if match[1] else 1
        equal = target >= 0 and target * target == coefficient * coefficient * int(match[2])
        return int(equal), "squared_radical"
    return None, "unsupported_expression"


def audit(directory):
    counts, mismatches, unsupported = collections.Counter(), [], []
    files = sorted((directory / "evaluations").glob("*.jsonl.gz"))
    if not files:
        raise ValueError("No portable validation responses to audit")
    for file in files:
        with gzip.open(file, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                expected, method = check_value(row["parsed_answer"], row["label"])
                counts["responses"] += 1
                counts[method] += 1
                details = {"file": file.name, "source_row": row["metadata"]["source_row"], "label": row["label"], "answer": row["parsed_answer"], "native_reward": row["reward"]}
                if expected is None:
                    unsupported.append(details)
                elif expected != row["reward"]:
                    mismatches.append({**details, "independent_reward": expected, "method": method})
    result = {"scope": "Independent exact arithmetic on native-extracted values. Does not independently validate box extraction or benchmark labels.", "validation_checkpoints": len(files), "counts": dict(counts), "mismatches": mismatches, "unsupported": unsupported, "rewards_changed": False}
    write_json(directory / "independent-numeric-audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    result = audit(parser.parse_args().directory)
    print(json.dumps(result, indent=2))
    if result["mismatches"]:
        raise SystemExit(1)
