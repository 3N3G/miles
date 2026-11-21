"""Fit transport lenses on unlabeled text and inspect frozen development readouts."""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

from pilot import MODEL, REVISION
from transport_math import algebra_checks, centered, fit_ridge, fit_rotation, plane_extrapolation, prediction_error, transport

VISITS = [11, 23, 35, 41]


def readout(prediction, head, norm_weight):
    h = prediction.float()
    h = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-6) * norm_weight
    return torch.tanh(torch.nn.functional.linear(h.bfloat16(), head).float() / 30) * 30


def score_predictions(prediction, native, items, tokens, semantic_mask, head, norm_weight, metadata):
    logits = readout(prediction, head, norm_weight)
    p, q = native.log_softmax(-1), logits.log_softmax(-1)
    kl = (p.exp() * (p - q)).sum(-1)
    rows = []
    for filtered in [False, True]:
        scored = logits.masked_fill(~semantic_mask, -float("inf")) if filtered else logits
        top_ids = scored.topk(10, dim=-1).indices.tolist()
        for i, item in enumerate(items):
            hit = bool(set(top_ids[i]) & set(item["target_ids"]))
            distractors = [name for name, ids in item["distractor_ids"].items() if set(top_ids[i]) & set(ids)]
            rows.append({**metadata, "name": item["item"]["name"], "filtered": filtered, "top10": [{"id": j, "text": tokens[j]} for j in top_ids[i]], "target_ids": item["target_ids"], "hit": hit, "distractors_read": distractors, "selective_hit": hit and not distractors, "has_single_token_distractors": bool(item["distractor_ids"]), "capable_unspoken": item["capable_unspoken"], "eligible": bool(item["target_ids"]), "native_top1_agrees": bool(logits[i].argmax() == native[i].argmax()), "output_kl": kl[i].item()})
    return rows


def prepare_items(outputs, tokens):
    token_map = {}
    for i, token in enumerate(tokens):
        token_map.setdefault(token.strip().casefold(), []).append(i)
    for row in outputs:
        item = row["item"]
        row["target_ids"] = sorted({i for c in item["intermediates"] for i in token_map.get(c.casefold(), [])})
        row["distractor_ids"] = {c: token_map[c.casefold()] for c in item.get("mc", {}).get("distractors", []) if c.casefold() in token_map}
        correct = bool(re.match(r"^[\s\"']*" + re.escape(item["target"]) + r"(?!\w)", row["generated_text"], flags=re.IGNORECASE))
        absent = all(not re.search(r"(?<!\w)" + re.escape(c) + r"(?!\w)", row["generated_text"], flags=re.IGNORECASE) for c in item["intermediates"])
        not_echo = all(not re.search(r"(?<!\w)" + re.escape(c) + r"(?!\w)", item["prompt"], flags=re.IGNORECASE) for c in item["intermediates"])
        row["capable_unspoken"] = correct and absent and not_echo and item["name"] != "firstletter-populous-country"
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert (args.capture / "capture_manifest.json").exists(), "Capture has not completed"
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(4)
    checks = algebra_checks("cuda")
    print(json.dumps({"event": "algebra_checks", **checks}), flush=True)
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
    target = torch.from_numpy(np.load(args.capture / "calibration_target.npy")).to("cuda")
    valid_indices = torch.arange(128 * 48, 144 * 48, device="cuda")
    yv = target[valid_indices]
    rows, diagnostics, maps = [], [], {}
    with torch.inference_mode():
        for mode in ["active", "mixed"]:
            cache = torch.from_numpy(np.load(args.capture / f"calibration_{mode}.npy")).to("cuda")
            development = torch.tensor(dev[mode], device="cuda")
            for seed in [0, 1, 2]:
                maps = {}
                paragraphs = np.random.default_rng(seed).choice(128, size=96, replace=False)
                flat = (paragraphs[:, None] * 48 + np.arange(48)[None, :]).reshape(-1)
                index = torch.tensor(flat, device="cuda")
                y = target[index]
                my, sy, _ = centered(y)
                for visit in VISITS:
                    step_start = time.monotonic()
                    x, xp = cache[index, visit].float(), cache[index, visit - 1].float()
                    xv, xpv = cache[valid_indices, visit].float(), cache[valid_indices, visit - 1].float()
                    xd, xpd = development[:, visit].float(), development[:, visit - 1].float()
                    mx, sx, _ = centered(x)
                    mp, sp, _ = centered(xp)
                    remaining = 47 - visit
                    predictions = {"raw": xd, "mean": my.expand_as(xd), "mean_scale": my + (xd - mx) * (sy / sx)}
                    validation = {"raw": xv, "mean": my.expand_as(xv), "mean_scale": my + (xv - mx) * (sy / sx)}
                    predictions["recent_plane_raw"] = plane_extrapolation(xpd, xd, remaining)
                    validation["recent_plane_raw"] = plane_extrapolation(xpv, xv, remaining)
                    damping_errors = []
                    for damping in [0, 0.25, 1]:
                        pred = plane_extrapolation((xpv - mp) / sp, (xv - mx) / sx, remaining * damping) * sy + my
                        damping_errors.append((prediction_error(pred, yv), damping))
                    _, damping = min(damping_errors)
                    predictions["recent_plane_centered"] = plane_extrapolation((xpd - mp) / sp, (xd - mx) / sx, remaining * damping) * sy + my
                    validation["recent_plane_centered"] = plane_extrapolation((xpv - mp) / sp, (xv - mx) / sx, remaining * damping) * sy + my
                    direct = fit_rotation(x, y)
                    predictions["direct_rotation"] = transport(xd, direct)
                    validation["direct_rotation"] = transport(xv, direct)
                    recent = fit_rotation(xp, x)
                    power = torch.linalg.matrix_power(recent["matrix"], remaining)
                    predictions["recent_rotation_power"] = (xd - mx) / sx @ power * sy + my
                    validation["recent_rotation_power"] = (xv - mx) / sx @ power * sy + my
                    ridge_options = []
                    for ridge in [0.01, 0.1, 1]:
                        fit = fit_ridge(x, y, ridge)
                        error = prediction_error(transport(xv, fit), yv)
                        ridge_options.append((error, ridge, fit))
                    ridge_error, ridge, ridge_fit = min(ridge_options, key=lambda value: value[0])
                    predictions["ridge"] = transport(xd, ridge_fit)
                    validation["ridge"] = transport(xv, ridge_fit)
                    # The shuffled-pair map is fitted on the same budget and cannot learn the true coupling.
                    generator = torch.Generator(device="cuda").manual_seed(seed)
                    permutation = torch.randperm(len(y), generator=generator, device="cuda")
                    shuffled = fit_ridge(x, y[permutation], ridge)
                    predictions["shuffled_ridge"] = transport(xd, shuffled)
                    validation["shuffled_ridge"] = transport(xv, shuffled)
                    for method, prediction in predictions.items():
                        meta = {"mode": mode, "seed": seed, "visit": visit, "method": method}
                        rows.extend(score_predictions(prediction, native, items, tokens, semantic_mask, head, norm_weight, meta))
                        per_paragraph = (validation[method] - yv).square().mean(-1).reshape(16, 48).mean(-1) / (yv - yv.mean(0)).square().mean()
                        cosine = torch.nn.functional.cosine_similarity(validation[method] - my, yv - my, dim=-1).reshape(16, 48).mean(-1)
                        diagnostic = {**meta, "validation_normalized_mse": prediction_error(validation[method], yv), "validation_normalized_mse_per_paragraph": per_paragraph.cpu().tolist(), "validation_centered_cosine_per_paragraph": cosine.cpu().tolist(), "damping": damping if method == "recent_plane_centered" else None, "ridge": ridge if method in ["ridge", "shuffled_ridge"] else None}
                        diagnostics.append(diagnostic)
                    for label, fit in [("direct_rotation", direct), ("ridge", ridge_fit)]:
                        prefix = f"{mode}_v{visit}_{label}_"
                        for key, value in fit.items():
                            maps[prefix + key] = value.cpu().numpy().astype(np.float32)
                    prefix = f"{mode}_v{visit}_recent_rotation_power_"
                    for key, value in {"mx": mx, "my": my, "sx": sx, "sy": sy, "matrix": power}.items():
                        maps[prefix + key] = value.cpu().numpy().astype(np.float32)
                    (args.out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
                    (args.out / "readouts.json").write_text(json.dumps(rows, indent=2) + "\n")
                    print(json.dumps({"event": "fit_site", "mode": mode, "seed": seed, "visit": visit, "damping": damping, "ridge": ridge, "ridge_validation_error": ridge_error, "site_seconds": time.monotonic() - step_start, "total_seconds": time.monotonic() - start}), flush=True)
                np.savez_compressed(args.out / f"maps_{mode}_seed_{seed}.npz", **maps)
            del cache
    (args.out / "fit_manifest.json").write_text(json.dumps({"model": MODEL, "revision": REVISION, "visits": VISITS, "fit_seeds": [0, 1, 2], "fit_paragraphs_per_seed": 96, "validation_paragraphs": 16, "parameters_selected_using": "unlabeled text geometric error only", "development_prompts": len(items), "algebra_checks": checks, "elapsed_seconds": time.monotonic() - start}, indent=2) + "\n")
    print(json.dumps({"event": "complete", "elapsed_seconds": time.monotonic() - start}), flush=True)


if __name__ == "__main__":
    main()
