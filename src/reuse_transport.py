"""Reuse pass-three decoding maps at earlier visits of the same physical layer."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

from fit_transport import VISITS, prepare_items, score_predictions
from pilot import MODEL, REVISION
from transport_math import prediction_error, transport


def load_fit(data, mode, visit, method, device="cuda"):
    prefix = f"{mode}_v{visit}_{method}_"
    return {key: torch.tensor(data[prefix + key], device=device, dtype=torch.float32) for key in ["mx", "sx", "my", "sy", "matrix"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    filename = hf_hub_download(MODEL, "model-00001-of-00001.safetensors", revision=REVISION)
    with safe_open(filename, framework="pt", device="cpu") as f:
        head = f.get_tensor("model.llm.unembed.weight")[:200058].to("cuda")
        norm_weight = f.get_tensor("model.llm.norm.weight").float().to("cuda")
    tokens = [tokenizer.decode([i]) for i in range(len(head))]
    semantic_mask = torch.tensor([any(c.isalpha() or c.isdigit() for c in t) for t in tokens], device="cuda")
    items = prepare_items(json.loads((args.capture / "development_outputs.json").read_text()), tokens)
    dev = np.load(args.capture / "development_states.npz")
    native = torch.tensor(dev["logits"], device="cuda")
    target = torch.from_numpy(np.load(args.capture / "calibration_target.npy")[128 * 48 :]).to("cuda")
    rows, diagnostics = [], []
    with torch.inference_mode():
        for mode in ["active", "mixed"]:
            cache = np.load(args.capture / f"calibration_{mode}.npy", mmap_mode="r")
            for seed in [0, 1, 2]:
                maps = np.load(args.fits / f"maps_{mode}_seed_{seed}.npz")
                for visit in VISITS:
                    late_visit = 35 if visit in [11, 23, 35] else 41
                    xd = torch.tensor(dev[mode][:, visit], device="cuda", dtype=torch.float32)
                    xv = torch.tensor(np.array(cache[128 * 48 :, visit]), device="cuda", dtype=torch.float32)
                    for method in ["direct_rotation", "ridge"]:
                        late = load_fit(maps, mode, late_visit, method)
                        own = load_fit(maps, mode, visit, method)
                        for recentered in [False, True]:
                            fit = {**late, "mx": own["mx"], "sx": own["sx"]} if recentered else late
                            label = f"late_reuse_{method}" + ("_recentered" if recentered else "")
                            meta = {"mode": mode, "seed": seed, "visit": visit, "method": label, "trained_visit": late_visit}
                            rows.extend(score_predictions(transport(xd, fit), native, items, tokens, semantic_mask, head, norm_weight, meta))
                            prediction = transport(xv, fit)
                            per_paragraph = (prediction - target).square().mean(-1).reshape(16, 48).mean(-1) / (target - target.mean(0)).square().mean()
                            diagnostics.append({**meta, "validation_normalized_mse": prediction_error(prediction, target), "validation_normalized_mse_per_paragraph": per_paragraph.cpu().tolist()})
                    print(json.dumps({"event": "reuse_site", "mode": mode, "seed": seed, "visit": visit, "trained_visit": late_visit}), flush=True)
    (args.out / "readouts.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    (args.out / "manifest.json").write_text(json.dumps({"model": MODEL, "revision": REVISION, "fit_seeds": [0, 1, 2], "development_prompts": len(items), "elapsed_seconds": time.monotonic() - start}, indent=2) + "\n")
    print(json.dumps({"event": "complete", "elapsed_seconds": time.monotonic() - start}), flush=True)


if __name__ == "__main__":
    main()
