"""Validate completed pilot artifacts and emit a compact evidence record."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_operator_pilot import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    read = lambda name: json.loads((args.run/name).read_text())
    manifest, controls, config = read("manifest.json"), read("native_controls.json"), read("config.json")
    inputs, fits, metrics = read("inputs.json"), read("fit_metadata.json"), read("final_metrics.json")
    assert manifest["code_commit"] == "c6ea41d9f71801d2bdee4fe8883425896ca66fe9"
    assert manifest["inputs_sha256"] == hashlib.sha256((args.run/"inputs.json").read_bytes()).hexdigest()
    assert manifest["config_sha256"] == hashlib.sha256((args.run/"config.json").read_bytes()).hexdigest()
    assert controls["operator_replay_max_errors"] == [0, 0, 0]
    assert controls["observational_hook_logit_error"] == 0
    assert controls["native_logits_unchanged_after_fitting"]
    assert not manifest["native_weights_updated"]
    assert len(fits) == 15 and len(metrics) == 54
    assert len(list(args.run.glob("routing_*.npz"))) == 176
    records = inputs["records"]
    assert len(records) == len({r["input_sha256"] for r in records}) == 176
    assert len({r["article"] for r in records if r["split"] == "final"}) == 32
    assert all(np.isfinite(r["per_paragraph_normalized_mse"]).all() and len(r["per_paragraph_normalized_mse"]) == 32 for r in metrics)
    for row in fits:
        assert row["epochs"] == 20 and len(row["history"]) == 20
        assert row["parameters"] == 18880768
        assert row["training_rows"] == (6144 if row["model"].startswith("pass") else 18432)
    summary = summarize(metrics)
    native, edits = read("native_replacements.json"), read("native_feature_edits.json")
    assert len(native) == 24 and len(edits) == 192
    feature_edits = [r for r in edits if r["direction"] == "feature"]
    lookup = {(r["id"], r["feature"], r["pass"], r["direction"]): r for r in edits}
    actual_norm_differences = []
    for row in feature_edits:
        random = lookup[(row["id"], row["feature"], row["pass"], "matched_norm_random")]
        assert np.isclose(row["intended_edit_norm"], random["intended_edit_norm"], rtol=1e-5, atol=1e-6)
        if row["intended_edit_norm"] == 0:
            assert row["last_token_kl"] == 0
            assert row["native_top10"] == row["edited_top10"]
        else:
            actual_norm_differences.append(abs(row["actual_bf16_edit_norm"]-random["actual_bf16_edit_norm"])/row["actual_bf16_edit_norm"])
    methods = {}
    for name in ["shared", "specific", "dense", "ridge", "specific_mean"]:
        chosen = []
        for p in [1, 2, 3]:
            match = lambda r: r["model"].startswith(f"pass{p}_seed") if name == "specific" else r["model"].startswith(name+"_seed") if name in ["shared", "dense"] else r["model"] == name
            chosen.append(np.mean([r["normalized_mse"] for r in metrics if r["evaluation_pass"] == p and match(r)]))
        methods[name] = {"per_pass_nmse": chosen, "mean_nmse": float(np.mean(chosen))}
    native_summary = {}
    for name in ["shared", "pass_specific", "pass_mean"]:
        selected = [r for r in native if r["condition"] == name]
        native_summary[name] = {key: float(np.mean([r[key] for r in selected])) for key in ["mean_kl_positions16_63", "top1_agreement_positions16_63", "next_token_nll_positions16_62"]}
    expert = read("native_expert_contexts.json")
    provenance = read("article_provenance.json")
    assert not provenance["final_shared_with_fit_or_validation"]
    first = next(r for r in records if r["split"] == "final")
    example = {"id": first["id"], "input": first["rendered_input"], "per_method_nmse": {}}
    for name in ["shared", "specific", "dense", "ridge", "specific_mean"]:
        chosen = []
        for p in [1, 2, 3]:
            match = lambda r: r["model"].startswith(f"pass{p}_seed") if name == "specific" else r["model"].startswith(name+"_seed") if name in ["shared", "dense"] else r["model"] == name
            chosen.extend(r["per_paragraph_normalized_mse"][0] for r in metrics if r["evaluation_pass"] == p and match(r))
        example["per_method_nmse"][name] = float(np.mean(chosen))
    result = {"validation_status":"passed", "execution_manifest":manifest, "native_controls":controls,
              "methods":methods, "primary_comparison":summary, "first_random_final_article_example":example,
              "native_replacement_development":native_summary,
              "feature_edits":{"candidate_removals":len(feature_edits), "nonzero_removals":sum(r["intended_edit_norm"] > 0 for r in feature_edits),
                 "top1_changes":sum(r["native_top10"][0]["id"] != r["edited_top10"][0]["id"] for r in feature_edits),
                 "maximum_relative_rounded_norm_mismatch":float(max(actual_norm_differences))},
              "native_expert105_development":[r for r in expert["cue_enrichment"] if r["split"] == "validation" and r["expert"] == 105],
              "native_expert105_intervals":expert["expert105_bootstrap"],
              "data_provenance":{k:v for k,v in provenance.items() if k != "article_by_calibration_id"},
              "scope":{"prediction_evaluation":"32 untouched final articles, three fit seeds",
                       "native_routing_analysis":"Post hoc, 16 development articles, lexical cues",
                       "native_expert_causal_ablation_performed":False}}
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"validation":"passed", "methods":methods,"first_example":example,"feature_edits":result["feature_edits"]},indent=2))


if __name__ == "__main__":
    main()
