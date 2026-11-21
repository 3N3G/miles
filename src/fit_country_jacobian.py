"""Extract exact country rows of a corpus/position-averaged Jacobian."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from pilot import MODEL, REVISION

COUNTRIES = ["France", "Japan", "Germany", "China"]
SITES = [35, 41]


def final_mean(llm, ids, insert=True):
    deltas = {visit: torch.zeros(2304, device="cuda", requires_grad=True) for visit in SITES}
    counter, targets = [0], []

    def source_hook(module, args, output):
        visit = counter[0]
        counter[0] += 1
        if insert and visit in deltas:
            mask = torch.zeros_like(output[..., :1])
            mask[:, 16:] = 1
            return (output.float() + mask * deltas[visit]).to(output.dtype)
        return None

    def target_hook(module, args, output):
        targets.append(output[0][:, 16:].mean((0, 1)))

    handles = [layer.register_forward_hook(source_hook) for layer in llm.layers]
    handles.append(llm.final_input.register_forward_hook(target_hook))
    try:
        llm(input_ids=ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    assert counter[0] == 48 and len(targets) == 1
    return targets[0], deltas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model, info = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map={"": "cuda:0"}, output_loading_info=True)
    model.eval().requires_grad_(False)
    assert not any(info[k] for k in ["missing_keys", "mismatched_keys", "error_msgs"])
    llm = model.model.language_model
    word_rows = {}
    for country in COUNTRIES:
        ids = tokenizer.encode(" " + country, add_special_tokens=False)
        assert len(ids) == 1
        word_rows[country] = model.lm_head.weight[ids[0]].float() * llm.norm.weight.float()
    calibration = json.loads(args.calibration.read_text())
    gradients = np.zeros((128, len(SITES), len(COUNTRIES), 2304), dtype=np.float32)
    for i, text in enumerate(calibration["texts"][:128]):
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :64].to("cuda")
        if i == 0:
            with torch.no_grad():
                reference, _ = final_mean(llm, ids, insert=False)
        target, deltas = final_mean(llm, ids)
        if i == 0:
            zero_error = (target.detach() - reference).abs().max().item()
            assert zero_error == 0, zero_error
            print(json.dumps({"event": "zero_insertion_control", "max_error": zero_error}), flush=True)
        for j, country in enumerate(COUNTRIES):
            score = target @ word_rows[country]
            values = torch.autograd.grad(score, [deltas[visit] for visit in SITES], retain_graph=j < len(COUNTRIES) - 1)
            for k, value in enumerate(values):
                assert torch.isfinite(value).all()
                gradients[i, k, j] = value.detach().float().cpu().numpy()
        if (i + 1) % 8 == 0:
            np.save(args.out / "paragraph_gradients.npy", gradients[: i + 1])
            print(json.dumps({"event": "gradient_paragraphs", "completed": i + 1, "elapsed_seconds": time.monotonic() - start}), flush=True)
        del target, deltas, values, score
    result, diagnostics = {}, []
    for seed in [0, 1, 2]:
        indices = np.random.default_rng(seed).choice(128, size=96, replace=False)
        means = gradients[indices].mean(0)
        for site_index, visit in enumerate(SITES):
            for word_index, country in enumerate(COUNTRIES):
                vector = means[site_index, word_index]
                result[f"seed_{seed}_v{visit}_{country}"] = vector
                norms = np.linalg.norm(gradients[indices, site_index, word_index], axis=-1)
                diagnostics.append({"seed": seed, "visit": visit, "country": country, "mean_vector_norm": float(np.linalg.norm(vector)), "mean_paragraph_norm": float(norms.mean()), "coherence_ratio": float(np.linalg.norm(vector) / norms.mean()), "paragraph_norm_max_to_median": float(norms.max() / np.median(norms))})
    np.savez_compressed(args.out / "country_rows.npz", **result)
    (args.out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    (args.out / "manifest.json").write_text(json.dumps({"model": MODEL, "revision": REVISION, "countries": COUNTRIES, "sites": SITES, "paragraphs": 128, "positions": [16, 63], "fit_seeds": [0, 1, 2], "paragraphs_per_seed": 96, "normalization": "True forward-model derivatives. Final RMS gain folded into vocabulary row; common final RMS denominator excluded as in hidden-state J-lens readout.", "elapsed_seconds": time.monotonic() - start}, indent=2) + "\n")
    print(json.dumps({"event": "complete", "elapsed_seconds": time.monotonic() - start}), flush=True)


if __name__ == "__main__":
    main()
