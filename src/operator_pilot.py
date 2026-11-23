"""Capture and fit a single shared Inkling sublayer inside an allocated GPU job."""

import argparse
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from operator_surrogate import (SparseOperator, document_metrics, fit_ridge,
                                fit_sparse, predict, predict_ridge)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def event(value):
    print(json.dumps(value), flush=True)


@torch.no_grad()
def capture(model, tokenizer, records, layer_index, out):
    operator = model.model.language_model.layers[layer_index].mlp
    device = next(model.parameters()).device
    xs, ys, routes, route_weights = [], [], [], []
    calls = []
    native_control = None

    def read_pairs(module, args, output):
        x = args[0]
        calls.append((x.detach().clone(), output.detach().clone()))

    def read_router(module, args, output):
        _, weights, indices = output
        routes.append(indices.reshape(-1, 64, 8)[:, 16:].cpu().numpy())
        route_weights.append(weights.reshape(-1, 64, 8)[:, 16:].cpu().numpy())

    pairs_handle = operator.register_forward_hook(read_pairs)
    router_handle = operator.gate.register_forward_hook(read_router)
    try:
        for index, record in enumerate(records):
            ids = tokenizer(record["text"], add_special_tokens=False).input_ids[:64]
            assert ids == record["token_ids"], "Tokenizer changed from frozen inputs"
            ids = torch.tensor([ids], device=device)
            calls.clear()
            routes.clear()
            route_weights.clear()
            logits = model(input_ids=ids, use_cache=False).logits
            assert len(calls) == len(routes) == 3
            if index == 0:
                native_control = logits.detach().clone()
                # Temporarily remove hooks so native replay does not alter the visit count.
                pairs_handle.remove()
                router_handle.remove()
                replay_errors = [(operator(x) - y).abs().max().item() for x, y in calls]
                assert replay_errors == [0, 0, 0], replay_errors
                untouched = model(input_ids=ids, use_cache=False).logits
                observation_error = (native_control - untouched).abs().max().item()
                assert observation_error == 0, observation_error
                pairs_handle = operator.register_forward_hook(read_pairs)
                router_handle = operator.gate.register_forward_hook(read_router)
                write(out / "native_controls.json", {"operator_replay_max_errors": replay_errors,
                      "observational_hook_logit_error": observation_error,
                      "first_input_logit_sha256": hashlib.sha256(native_control.float().cpu().numpy().tobytes()).hexdigest()})
            xs.append(np.stack([x[:, 16:].float().cpu().numpy()[0] for x, _ in calls], axis=1))
            ys.append(np.stack([y[:, 16:].float().cpu().numpy()[0] for _, y in calls], axis=1))
            # Each record contributes (position, pass, selected-expert).
            np.savez_compressed(out / f"routing_{index:03}.npz",
                                indices=np.stack(routes, axis=2)[0], weights=np.stack(route_weights, axis=2)[0])
            event({"event": "capture", "index": index, "id": record["id"], "split": record["split"]})
    finally:
        pairs_handle.remove()
        router_handle.remove()
    x, y = np.stack(xs), np.stack(ys)
    assert x.shape == y.shape == (len(records), 48, 3, 2304)
    # FP32 holds the native BF16 values exactly, with no extra rounding.
    np.save(out / "operator_inputs.npy", x)
    np.save(out / "operator_outputs.npy", y)
    return x, y


def split_tensors(x, y, records, split, device="cuda"):
    selected = [i for i, record in enumerate(records) if record["split"] == split]
    return (torch.from_numpy(x[selected]).to(device).reshape(-1, 3, x.shape[-1]),
            torch.from_numpy(y[selected]).to(device).reshape(-1, 3, y.shape[-1]))


def fit_all(x, y, records, config, out):
    device = config.get("device", "cuda")
    xf, yf = split_tensors(x, y, records, "fit", device)
    xv, yv = split_tensors(x, y, records, "validation", device)
    metadata = []
    for seed in config["seeds"]:
        specifications = [("shared", None, config["topk"]),
                          ("dense", None, config["width"])]
        specifications += [(f"pass{p+1}", p, config["topk"]) for p in range(3)]
        for name, pass_index, topk in specifications:
            select = lambda value: value.reshape(-1, value.shape[-1]) if pass_index is None else value[:, pass_index]
            key = f"{name}_seed{seed}"
            report = lambda row, key=key: event({"event": "training", "model": key, **row})
            checkpoint, details = fit_sparse(select(xf), select(yf), select(xv), select(yv),
                                            seed=seed, width=config["width"], topk=topk,
                                            epochs=config["epochs"], batch_size=config["batch_size"],
                                            learning_rate=config["learning_rate"], report=report)
            torch.save(checkpoint, out / f"{key}.pt")
            metadata.append({"model": key, **details})
            write(out / "fit_metadata.json", metadata)
            gc.collect()
            torch.cuda.empty_cache()
    ridge_options = []
    flat_xf, flat_yf = xf.reshape(-1, xf.shape[-1]), yf.reshape(-1, yf.shape[-1])
    for regularizer in config["ridge_regularizers"]:
        fit = fit_ridge(flat_xf, flat_yf, regularizer)
        error = (predict_ridge(xv.reshape(-1, xv.shape[-1]), fit) - yv.reshape(-1, yv.shape[-1])).square().mean().item()
        ridge_options.append((error, regularizer, {k: v.cpu() for k, v in fit.items()}))
    error, regularizer, ridge = min(ridge_options, key=lambda row: row[0])
    torch.save(ridge, out / "ridge.pt")
    means = {"shared": flat_yf.mean(0).cpu(), **{f"pass{p+1}": yf[:, p].mean(0).cpu() for p in range(3)}}
    torch.save(means, out / "means.pt")
    frozen = {"models": metadata, "ridge_regularizer": regularizer,
              "ridge_validation_mse": error, "final_outputs_used_for_selection": False}
    write(out / "frozen_fit.json", frozen)
    del xf, yf, xv, yv
    torch.cuda.empty_cache()


@torch.no_grad()
def evaluate_final(x, y, records, config, out):
    assert (out / "frozen_fit.json").exists()
    device = config.get("device", "cuda")
    xt, yt = split_tensors(x, y, records, "final", device)
    means = torch.load(out / "means.pt", weights_only=True)
    rows = []
    for path in sorted(out.glob("*_seed*.pt")):
        model = SparseOperator.from_checkpoint(torch.load(path, weights_only=True), device)
        for p in range(3):
            result = document_metrics(predict(model, xt[:, p]), yt[:, p], positions=48,
                                      training_mean=means[f"pass{p+1}"].to(device))
            rows.append({"model": path.stem, "evaluation_pass": p + 1, **result})
        del model
    ridge = {k: v.to(device) for k, v in torch.load(out / "ridge.pt", weights_only=True).items()}
    for p in range(3):
        baseline_predictions = {"shared_mean": means["shared"].to(device).expand_as(yt[:, p]),
                                "specific_mean": means[f"pass{p+1}"].to(device).expand_as(yt[:, p]),
                                "ridge": predict_ridge(xt[:, p], ridge)}
        for name, prediction in baseline_predictions.items():
            rows.append({"model": name, "evaluation_pass": p+1,
                         **document_metrics(prediction, yt[:, p], positions=48,
                                            training_mean=means[f"pass{p+1}"].to(device))})
    write(out / "final_metrics.json", rows)
    return rows


@torch.no_grad()
def feature_examples(x, records, tokenizer, config, out):
    model = SparseOperator.from_checkpoint(torch.load(out / "shared_seed0.pt", weights_only=True), "cuda")
    fit_indices = [i for i, r in enumerate(records) if r["split"] == "fit"]
    train = torch.from_numpy(x[fit_indices]).to("cuda").reshape(-1, 3, 2304)
    profiles, examples = [], []
    for p in range(3):
        z = torch.cat([model.encode(batch) for batch in train[:, p].split(256)])
        profiles.append({"pass": p+1, "firing_rate": (z > 0).float().mean(0).cpu().tolist(),
                         "mean_activation": z.mean(0).cpu().tolist(),
                         "mean_active_features": (z > 0).sum(-1).float().mean().item()})
        values, indices = z.topk(4, dim=0)
        for feature in range(model.width):
            contexts = []
            for rank in range(4):
                flat = int(indices[rank, feature])
                document, position = fit_indices[flat // 48], flat % 48 + 16
                ids = records[document]["token_ids"]
                contexts.append({"id": records[document]["id"], "position": position,
                                 "activation": float(values[rank, feature]),
                                 "token": tokenizer.decode([ids[position]]),
                                 "context": tokenizer.decode(ids[max(0, position-8):position+1])})
            examples.append({"feature": feature, "pass": p+1, "contexts": contexts})
    write(out / "feature_profiles.json", profiles)
    write(out / "feature_examples.json", examples)
    del train
    return model


def main():
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from pilot import MODEL, REVISION

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["allocation_budget_status"] != "approved":
        raise PermissionError("The project's required allocation-budget decision is pending")
    inputs = json.loads(args.inputs.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(4)
    assert REVISION == inputs["model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model, loading = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION,
                        dtype=torch.bfloat16, device_map={"": "cuda:0"}, output_loading_info=True)
    model.eval().requires_grad_(False)
    assert not any(loading[k] for k in ["missing_keys", "mismatched_keys", "error_msgs"])
    assert model.model.language_model.config.loop_num_passes == 3
    for split in ["fit", "validation", "final"]:
        first = next(r for r in inputs["records"] if r["split"] == split)
        event({"event": "rendered_input", "split": split, **first})
    write(args.out / "config.json", config)
    write(args.out / "inputs.json", inputs)
    x, y = capture(model, tokenizer, inputs["records"], config["layer_index"], args.out)
    # The frozen native model remains on the GPU, but receives no gradients.
    fit_all(x, y, inputs["records"], config, args.out)
    evaluate_final(x, y, inputs["records"], config, args.out)
    surrogate = feature_examples(x, inputs["records"], tokenizer, config, args.out)
    from operator_native_checks import native_checks
    native_checks(model, tokenizer, inputs["records"], surrogate, config, args.out)
    manifest = {"model": MODEL, "revision": REVISION, "torch": torch.__version__,
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "inputs_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "elapsed_seconds": time.monotonic()-start,
                "maximum_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                "surrogate_fitting_only": True, "native_weights_updated": False}
    write(args.out / "manifest.json", manifest)
    event({"event": "complete", **manifest})


if __name__ == "__main__":
    main()
