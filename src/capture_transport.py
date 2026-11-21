"""Capture unlabeled trajectories and frozen development prompts for transport lenses."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from pilot import MODEL, REVISION, token_list


def capture(model, ids, positions, verify=False):
    llm = model.model.language_model
    active, mixed, final = [], [], []

    def read_layer(module, args, output):
        entering, completed = args[:2]
        states = completed[:, positions]
        if module.is_block_boundary:
            states = torch.cat([states, entering[:, positions].unsqueeze(-2)], dim=-2)
        partial = output[:, positions]
        active.append(partial.detach().half())
        # Calling forward directly avoids triggering the native-final-reader hook below.
        mixed.append(llm.final_input.forward(states, partial)[0].detach().half())

    def read_final(module, args, output):
        final.append(output[0])

    handles = [layer.register_forward_hook(read_layer) for layer in llm.layers]
    handles.append(llm.final_input.register_forward_hook(read_final))
    try:
        output = model(input_ids=ids, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    assert len(active) == len(mixed) == 48
    assert len(final) == 1
    if verify:
        reconstructed = model.logit_softcap(model.lm_head(llm.norm(final[0]).to(model.lm_head.weight.dtype)).float())
        error = (reconstructed - output.logits).abs().max().item()
        assert error == 0, error
    else:
        error = None
    # Flatten batch and selected token positions, preserving the visit axis.
    arrays = {name: torch.stack(values, dim=2).cpu().numpy().reshape(-1, 48, llm.config.hidden_size) for name, values in [("active", active), ("mixed", mixed)]}
    target = final[0][:, positions].float().cpu().numpy().reshape(-1, llm.config.hidden_size)
    return arrays, target, output.logits[:, -1].float().cpu().numpy(), error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model, load_info = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map={"": "cuda:0"}, output_loading_info=True)
    model.eval().requires_grad_(False)
    assert not any(load_info[k] for k in ["missing_keys", "mismatched_keys", "error_msgs"])
    calibration = json.loads(args.calibration.read_text())
    token_rows = [tokenizer(text, add_special_tokens=False).input_ids[:64] for text in calibration["texts"]]
    assert len(token_rows) == 144 and all(len(x) == 64 for x in token_rows)
    positions = list(range(16, 64))
    n_rows = len(token_rows) * len(positions)
    caches = {name: np.lib.format.open_memmap(args.out / f"calibration_{name}.npy", mode="w+", dtype=np.float16, shape=(n_rows, 48, 2304)) for name in ["active", "mixed"]}
    targets = np.lib.format.open_memmap(args.out / "calibration_target.npy", mode="w+", dtype=np.float32, shape=(n_rows, 2304))
    inputs = {"model": MODEL, "revision": REVISION, "render": "plain text, no added special tokens", "fit_paragraphs": 128, "validation_paragraphs": 16, "selected_positions": positions, "token_rows": token_rows, "first_rendered_input": tokenizer.decode(token_rows[0])}
    (args.out / "calibration_inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    print(json.dumps({"event": "first_calibration_input", "text": inputs["first_rendered_input"], "token_ids": token_rows[0]}), flush=True)
    with torch.inference_mode():
        for begin in range(0, len(token_rows), 4):
            ids = torch.tensor(token_rows[begin : begin + 4], device="cuda")
            data, target, logits, error = capture(model, ids, positions, verify=begin == 0)
            left, right = begin * len(positions), (begin + len(ids)) * len(positions)
            for name in caches:
                caches[name][left:right] = data[name]
            targets[left:right] = target
            print(json.dumps({"event": "calibration_batch", "paragraphs_completed": begin + len(ids), "native_logit_error": error, "elapsed": time.monotonic() - start}), flush=True)
        for value in [*caches.values(), targets]:
            value.flush()
        items = json.loads(args.development.read_text())["items"]
        dev = {name: [] for name in ["active", "mixed", "target", "logits"]}
        rows = []
        for item in items:
            ids = tokenizer(item["prompt"], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            assert ids.shape[-1] <= 256
            data, target, logits, error = capture(model, ids, [-1], verify=True)
            for name in ["active", "mixed"]:
                dev[name].append(data[name][0])
            dev["target"].append(target[0])
            dev["logits"].append(logits[0])
            generated = []
            next_logits = torch.tensor(logits[0], device="cuda")
            for step in range(12):
                if step:
                    next_logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
                token = int(next_logits.argmax())
                generated.append(token)
                if token == 200006:
                    break
                ids = torch.cat([ids, ids.new_tensor([[token]])], dim=-1)
            row = {"item": item, "native_logit_error": error, "final_tokens": token_list(torch.tensor(logits[0]), tokenizer), "generated_ids": generated, "generated_text": tokenizer.decode(generated)}
            rows.append(row)
            (args.out / "development_outputs.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps({"event": "development", "name": item["name"], "prompt": item["prompt"], "generated_text": row["generated_text"], "native_logit_error": error}), flush=True)
        np.savez_compressed(args.out / "development_states.npz", **{name: np.array(values) for name, values in dev.items()})
    manifest = {"model": MODEL, "revision": REVISION, "paragraphs": len(token_rows), "positions_per_paragraph": len(positions), "development_prompts": len(rows), "cache_dtype": "FP16 for source trajectories, FP32 for final target", "elapsed_seconds": time.monotonic() - start, "state_layout": "token_row, visit, hidden_coordinate", "visits": list(range(48))}
    (args.out / "capture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", **manifest}), flush=True)


if __name__ == "__main__":
    main()
