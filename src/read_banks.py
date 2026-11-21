"""Decode saved banks with the released output weights. No model execution or fitting."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

from pilot import MODEL, REVISION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    filename = hf_hub_download(MODEL, "model-00001-of-00001.safetensors", revision=REVISION)
    with safe_open(filename, framework="pt", device="cpu") as f:
        head = f.get_tensor("model.llm.unembed.weight")[:200058].to("cuda")
        norm_weight = f.get_tensor("model.llm.norm.weight").float().to("cuda")
    controls = sorted(args.run.glob("control-*.npz"))
    assert len(controls) == 4
    normalized = lambda x: x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    mean = np.mean([normalized(np.load(p)["banks"]) for p in controls], axis=0)
    tokens = [tokenizer.decode([i]) for i in range(head.shape[0])]
    semantic_mask = torch.tensor([any(c.isalpha() or c.isdigit() for c in t) for t in tokens], device="cuda")
    rows = []
    for path in sorted(args.run.glob("*.npz")):
        if path.name == "gate_weights.npz":
            continue
        item = json.loads(path.with_suffix(".json").read_text())["item"]
        bank = normalized(np.load(path)["banks"])
        variants = {"raw": bank, "centered": bank - mean, "delta": bank[1:] - bank[:-1], "centered_delta": (bank - mean)[1:] - (bank - mean)[:-1]}
        for method, states in variants.items():
            h = torch.tensor(states, device="cuda", dtype=torch.float32)
            h = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-6)
            with torch.inference_mode():
                logits = torch.nn.functional.linear((h * norm_weight).bfloat16(), head).float()
            target_ids = sorted({i for concept in item["intermediates"] for i, text in enumerate(tokens) if text.strip().casefold() == concept.casefold()})
            for filtered in [False, True]:
                scored = logits.masked_fill(~semantic_mask, -float("inf")) if filtered else logits
                for t in range(states.shape[0]):
                    for b in range(2):
                        values, ids = scored[t, b].topk(10)
                        ranks = {str(i): int((scored[t, b] > scored[t, b, i]).sum()) + 1 for i in target_ids}
                        # Ties receive the best rank, but a hit requires membership in the actual top-K list.
                        rows.append({"name": item["name"], "prompt": item["prompt"], "intermediates": item["intermediates"], "method": method, "filtered": filtered, "loop": t + (2 if "delta" in method else 1), "bank": b, "target_ids": target_ids, "target_ranks": ranks, "top10": [{"id": i, "text": tokens[i], "score": v} for i, v in zip(ids.tolist(), values.tolist())], "hit": bool(set(ids.tolist()) & set(target_ids))})
    args.out.write_text(json.dumps({"model": MODEL, "revision": REVISION, "calibration": [p.stem for p in controls], "calibration_rule": "Mean normalized bank per loop and block on four fixed engineering controls. No target labels used.", "evaluation": "12 exploratory development items, plus controls shown separately. Not held-out performance.", "rows": rows}, indent=2) + "\n")
    for method in ["raw", "centered", "delta", "centered_delta"]:
        for filtered in [False, True]:
            # Use the same two loop horizons for every method.
            selected = [r for r in rows if r["method"] == method and r["filtered"] == filtered and r["loop"] >= 2 and not r["name"].startswith("control-")]
            names = sorted({r["name"] for r in selected if r["target_ids"]})
            hits = [name for name in names if any(r["hit"] for r in selected if r["name"] == name)]
            print(json.dumps({"method": method, "filtered": filtered, "eligible": len(names), "hits": hits}), flush=True)


if __name__ == "__main__":
    main()
