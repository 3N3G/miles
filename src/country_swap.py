"""Test whether country readout directions support cross-task causal exchanges."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from fit_transport import VISITS
from pilot import MODEL, REVISION, token_list


def features(model, tokenizer, fits, method, seed):
    source = tokenizer.encode(" France", add_special_tokens=False)
    target = tokenizer.encode(" Japan", add_special_tokens=False)
    assert len(source) == len(target) == 1
    contrast = (model.lm_head.weight[source[0]].float() - model.lm_head.weight[target[0]].float()) * model.model.language_model.norm.weight.float()
    maps = np.load(fits / f"maps_active_seed_{seed}.npz")
    result = {}
    for visit in VISITS:
        prefix = f"active_v{visit}_ridge_"
        stats = {k: torch.tensor(maps[prefix + k], device="cuda", dtype=torch.float32) for k in ["mx", "sx", "my", "sy"]}
        if method == "raw":
            vector, bias = contrast, contrast.new_tensor(0)
        elif method == "mean_scale":
            vector = contrast * stats["sy"] / stats["sx"]
            bias = stats["my"] @ contrast - stats["mx"] @ vector
        elif method == "ridge":
            matrix = torch.tensor(maps[prefix + "matrix"], device="cuda", dtype=torch.float32)
            vector = (matrix @ contrast) * stats["sy"] / stats["sx"]
            bias = stats["my"] @ contrast - stats["mx"] @ vector
        else:
            raise ValueError(method)
        assert torch.isfinite(vector).all() and vector.square().sum() > 1e-12
        result[visit] = {"vector": vector, "bias": bias}
    return result


def forward_with_swap(model, ids, read_features=None, dose=0, random_seed=None):
    counter, changes = [0], []

    def hook(module, args, output):
        visit = counter[0]
        counter[0] += 1
        if read_features is None or visit not in read_features:
            return None
        vector, bias = read_features[visit]["vector"], read_features[visit]["bias"]
        h = output.float()
        score = h @ vector + bias
        delta = -2 * score.unsqueeze(-1) * vector / vector.square().sum()
        if random_seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(1000 * random_seed + visit)
            random_direction = torch.randn(vector.shape, generator=generator, device="cuda")
            random_direction /= random_direction.norm()
            delta = delta.norm(dim=-1, keepdim=True) * random_direction
        delta *= dose
        changed = (h + delta).to(output.dtype)
        after = changed.float() @ vector + bias
        changes.append({"visit": visit, "contrast_before_last": score[0, -1].item(), "contrast_after_last": after[0, -1].item(), "perturbation_relative_norm_last": (delta[0, -1].norm() / h[0, -1].norm().clamp_min(1e-8)).item(), "perturbation_frobenius_norm": delta.norm().item(), "max_relative_reflection_rounding_error": ((after - (1 - 2 * dose) * score).abs() / (score.abs() + 1)).max().item() if random_seed is None else None})
        return output if dose == 0 else changed

    handles = [layer.register_forward_hook(hook) for layer in model.model.language_model.layers]
    try:
        logits = model(input_ids=ids, use_cache=False).logits[0, -1].float().clone()
    finally:
        for handle in handles:
            handle.remove()
    assert counter[0] == 48
    assert torch.isfinite(logits).all()
    return logits, changes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model, loading = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map={"": "cuda:0"}, output_loading_info=True)
    model.eval().requires_grad_(False)
    assert not any(loading[k] for k in ["missing_keys", "mismatched_keys", "error_msgs"])
    cases = json.loads(args.cases.read_text())["cases"]
    token_map = {}
    for i in range(model.lm_head.weight.shape[0]):
        token_map.setdefault(tokenizer.decode([i]).strip().casefold(), []).append(i)
    native, controls = {}, []
    rows = []
    with torch.inference_mode():
        for case in cases:
            for prompt in [case["prompt"], case.get("counterfactual_prompt")]:
                if prompt is None or prompt in native:
                    continue
                ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
                logits, _ = forward_with_swap(model, ids)
                native[prompt] = logits
            controls.append({"case": case, "original_top10": token_list(native[case["prompt"]], tokenizer), "counterfactual_input_top10": token_list(native[case["counterfactual_prompt"]], tokenizer) if case.get("counterfactual_prompt") else None})
        (args.out / "native_controls.json").write_text(json.dumps(controls, indent=2) + "\n")
        for method in ["mean_scale", "raw", "ridge"]:
            for seed in [0, 1, 2]:
                read_features = features(model, tokenizer, args.fits, method, seed)
                for case in cases:
                    ids = tokenizer(case["prompt"], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
                    base = native[case["prompt"]]
                    source_ids = token_map[case["original"].casefold()]
                    target_ids = token_map[case["counterfactual"].casefold()]
                    baseline_valid = int(base.argmax()) in source_ids
                    input_control_valid = int(native[case["counterfactual_prompt"]].argmax()) in target_ids if case.get("counterfactual_prompt") else None
                    conditions = [(0, None), (0.5, None), (1, None), (2, None)] + [(1, r) for r in [0, 1, 2]]
                    for dose, random_seed in conditions:
                        logits, changes = forward_with_swap(model, ids, read_features, dose, random_seed)
                        if dose == 0:
                            assert (logits - base).abs().max().item() == 0
                        lp = logits.log_softmax(-1)
                        source_prob = torch.logsumexp(lp[source_ids], 0).exp().item()
                        target_prob = torch.logsumexp(lp[target_ids], 0).exp().item()
                        row = {"method": method, "fit_seed": seed, "name": case["name"], "kind": case["kind"], "prompt": case["prompt"], "dose": dose, "random_seed": random_seed, "baseline_valid": baseline_valid, "counterfactual_input_valid": input_control_valid, "source_probability": source_prob, "target_probability": target_prob, "counterfactual_top1": int(logits.argmax()) in target_ids, "unchanged_top1": int(logits.argmax()) == int(base.argmax()), "top5": token_list(logits, tokenizer, k=5), "changes": changes}
                        rows.append(row)
                        print(json.dumps({k: v for k, v in row.items() if k not in ["changes", "prompt"]}), flush=True)
                    (args.out / "interventions.json").write_text(json.dumps(rows, indent=2) + "\n")
                print(json.dumps({"event": "fit_seed_complete", "method": method, "seed": seed, "elapsed_seconds": time.monotonic() - start}), flush=True)
    (args.out / "manifest.json").write_text(json.dumps({"model": MODEL, "revision": REVISION, "visits": VISITS, "methods": ["mean_scale", "raw", "ridge"], "fit_seeds": [0, 1, 2], "random_control_seeds": [0, 1, 2], "cases": len(cases), "elapsed_seconds": time.monotonic() - start}, indent=2) + "\n")
    print(json.dumps({"event": "complete", "elapsed_seconds": time.monotonic() - start}), flush=True)


if __name__ == "__main__":
    main()
