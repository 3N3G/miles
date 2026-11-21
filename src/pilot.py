"""Inspect the released model without training it. Run inside an allocated GPU job."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL = "thinkingmachines/Inkling-2-Mini"
REVISION = "ed40e7da625f594fc9c4990f5f66be24873c22d7"


def rms(x):
    return x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)


def token_list(logits, tokenizer, k=10):
    values, ids = logits.float().topk(k)
    return [
        {"id": i, "text": tokenizer.decode([i]), "logit": v}
        for i, v in zip(ids.tolist(), values.tolist())
    ]


def decode(model, llm, tokenizer, h):
    h = llm.norm(h).to(model.lm_head.weight.dtype)
    logits = model.logit_softcap(model.lm_head(h)).float()
    return token_list(logits, tokenizer)


def angular_change(x, seed, angle):
    generator = torch.Generator(device=x.device).manual_seed(seed)
    x = x.float()
    noise = torch.randn(x.shape, generator=generator, device=x.device)
    noise -= (noise * x).sum(-1, keepdim=True) / x.square().sum(-1, keepdim=True).clamp_min(1e-12) * x
    noise *= x.norm(dim=-1, keepdim=True) / noise.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return math.cos(angle) * x + math.sin(angle) * noise


def intervention_hook(kind, bank, value, seed):
    def hook(module, args):
        old, active, layer_idx = args
        if kind == "scale":
            if bank == 0:
                old = old * value
            else:
                active = active * value
        elif kind == "angle":
            # A rotation by pi/3 has displacement ||x||, matching a scale of 2.
            if bank == 0:
                old = angular_change(old, seed, value)
            else:
                active = angular_change(active, seed, value)
        return old, active, layer_idx

    return hook


def baseline(model, llm, tokenizer, ids):
    handles, readouts, banks, visits = [], [], [], {}
    final_mixed = []

    def read_layer(layer_idx):
        def hook(module, args, output):
            count = visits.get(layer_idx, 0)
            visits[layer_idx] = count + 1
            readouts.append({"layer": layer_idx, "pass": count + 1, "site": "active", "tokens": decode(model, llm, tokenizer, output[0, -1])})

        return hook

    def read_bank(module, args):
        old, active, layer_idx = args
        if layer_idx == 0:
            bank = torch.cat([old, active.unsqueeze(-2)], dim=-2)
            banks.append(bank[0, -1].float().cpu())

    def read_coda(module, args):
        active, completed = args[:2]
        banks.append(torch.stack([completed[0, -1, -1], active[0, -1]]).float().cpu())

    def read_final(module, args, output):
        final_mixed.append(output[0][0, -1].clone())

    for i, layer in enumerate(llm.layers):
        handles.append(layer.register_forward_hook(read_layer(i)))
    handles.append(llm.message_input.register_forward_pre_hook(read_bank))
    handles.append(llm.layers[18].register_forward_pre_hook(read_coda))
    handles.append(llm.final_input.register_forward_hook(read_final))
    try:
        output = model(input_ids=ids, use_cache=False).logits[0, -1].float()
    finally:
        for handle in handles:
            handle.remove()
    assert len(readouts) == 48, len(readouts)
    assert len(banks) == 3, [x.shape for x in banks]
    expected = model.logit_softcap(model.lm_head(llm.norm(final_mixed[0]).to(model.lm_head.weight.dtype))).float()
    error = (expected - output).abs().max().item()
    if error > 1e-4 or not torch.isfinite(output).all():
        raise RuntimeError(f"Native-logit control failed: max error {error}")
    bank_array = torch.stack(banks)
    normalized = rms(bank_array)
    delta_readouts = []
    for loop in range(1, 3):
        for bank_idx in range(2):
            delta = (normalized[loop, bank_idx] - normalized[loop - 1, bank_idx]).to(ids.device)
            delta_readouts.append({"loop": loop + 1, "bank": bank_idx, "tokens": decode(model, llm, tokenizer, delta)})
    raw_distance = (bank_array[1:] - bank_array[:-1]).norm(dim=-1).tolist()
    normalized_distance = (normalized[1:] - normalized[:-1]).norm(dim=-1).tolist()
    return output, bank_array.numpy(), {"readouts": readouts, "delta_readouts": delta_readouts, "raw_distance": raw_distance, "normalized_distance": normalized_distance, "native_logit_error": error}


def compare(base, changed):
    lp, lq = base.log_softmax(-1), changed.log_softmax(-1)
    return {
        "kl_base_to_changed": (lp.exp() * (lp - lq)).sum().item(),
        "max_logit_change": (base - changed).abs().max().item(),
        "top1_agrees": base.argmax().item() == changed.argmax().item(),
    }


def run_prompt(model, llm, tokenizer, item, out, do_interventions):
    prompt = item["prompt"]
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    if ids.shape[-1] > 256:
        raise ValueError("Prompt exceeds the pilot's 256-token bound")
    start = time.monotonic()
    print(json.dumps({"event": "input", "name": item["name"], "render": "plain, verbatim, no chat template", "prompt": prompt, "token_ids": ids[0].tolist()}), flush=True)
    base, bank_array, result = baseline(model, llm, tokenizer, ids)
    result.update({"item": item, "token_ids": ids[0].tolist(), "final_tokens": token_list(base, tokenizer), "interventions": []})
    print(json.dumps({"event": "baseline", "name": item["name"], "seconds": time.monotonic() - start, "top_tokens": result["final_tokens"]}), flush=True)
    if do_interventions:
        configs = [("scale", b, scale, 0) for b in range(2) for scale in [0.5, 2.0]]
        configs += [("angle", b, math.pi / 3, seed) for b in range(2) for seed in [0, 1, 2]]
        configs += [("scale", 0, 1.0, 0)]
        for kind, bank, value, seed in configs:
            handle = llm.message_input.register_forward_pre_hook(intervention_hook(kind, bank, value, seed))
            try:
                changed = model(input_ids=ids, use_cache=False).logits[0, -1].float()
            finally:
                handle.remove()
            row = {"kind": kind, "bank": bank, "value": value, "seed": seed, **compare(base, changed), "top_tokens": token_list(changed, tokenizer, k=3)}
            result["interventions"].append(row)
            print(json.dumps({"event": "intervention", "name": item["name"], **row}), flush=True)
    np.savez_compressed(out / f"{item['name']}.npz", banks=bank_array, logits=base.cpu().numpy())
    result["elapsed_seconds"] = time.monotonic() - start
    (out / f"{item['name']}.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--max-prompts", type=int, default=16)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    random.seed(0)
    print(json.dumps({"event": "runtime", "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "model": MODEL, "revision": REVISION}), flush=True)
    start = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model, info = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map={"": "cuda:0"}, output_loading_info=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    (args.out / "loading_info.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
    # No missing text-model parameters or mismatched shapes may be ignored.
    errors = {k: v for k, v in info.items() if v and k in ["missing_keys", "mismatched_keys", "error_msgs"]}
    if errors:
        raise RuntimeError(f"Incomplete model load: {errors}")
    llm = model.model.language_model
    print(json.dumps({"event": "loaded", "seconds": time.monotonic() - start, "parameters": sum(p.numel() for p in model.parameters())}), flush=True)
    items = json.loads(args.items.read_text())["items"][: args.max_prompts]
    with torch.inference_mode():
        for i, item in enumerate(items):
            run_prompt(model, llm, tokenizer, item, args.out, do_interventions=i < 4)
    print(json.dumps({"event": "complete", "prompts": len(items), "seconds": time.monotonic() - start}), flush=True)


if __name__ == "__main__":
    main()
