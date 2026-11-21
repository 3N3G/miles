"""Summarize frozen-budget development readouts without selecting a test result."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows, diagnostics = [], []
    for run in args.runs:
        rows.extend(json.loads((run / "readouts.json").read_text()))
        diagnostics.extend(json.loads((run / "diagnostics.json").read_text()))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["mode"], row["method"], row["seed"], row["filtered"])].append(row)
    summaries = []
    for (mode, method, seed, filtered), selected in grouped.items():
        names = sorted({r["name"] for r in selected if r["eligible"]})
        capable = sorted({r["name"] for r in selected if r["eligible"] and r["capable_unspoken"]})
        hits = [n for n in names if any(r["hit"] for r in selected if r["name"] == n)]
        capable_hits = [n for n in capable if any(r["hit"] for r in selected if r["name"] == n)]
        summaries.append({"mode": mode, "method": method, "seed": seed, "filtered": filtered, "eligible": len(names), "hits": len(hits), "hit_names": hits, "capable_eligible": len(capable), "capable_hits": len(capable_hits), "capable_hit_names": capable_hits, "mean_output_kl": float(np.mean([r["output_kl"] for r in selected]))})
    averages = []
    keys = sorted({(r["mode"], r["method"], r["filtered"]) for r in summaries})
    for mode, method, filtered in keys:
        selected = [r for r in summaries if (r["mode"], r["method"], r["filtered"]) == (mode, method, filtered)]
        geo = [r["validation_normalized_mse"] for r in diagnostics if r["mode"] == mode and r["method"] == method]
        averages.append({"mode": mode, "method": method, "filtered": filtered, "seeds": [r["seed"] for r in selected], "hits_by_seed": [r["hits"] for r in selected], "eligible": selected[0]["eligible"], "capable_hits_by_seed": [r["capable_hits"] for r in selected], "capable_eligible": selected[0]["capable_eligible"], "mean_output_kl": float(np.mean([r["mean_output_kl"] for r in selected])), "mean_validation_normalized_mse": float(np.mean(geo)) if geo else None})
    result = {"scope": "Development only. Four fixed visits, one token position, K=10. No reserved test items evaluated.", "averages": averages, "per_seed": summaries}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for row in sorted([r for r in averages if not r["filtered"]], key=lambda r: -np.mean(r["capable_hits_by_seed"])):
        print(json.dumps(row))


if __name__ == "__main__":
    main()
