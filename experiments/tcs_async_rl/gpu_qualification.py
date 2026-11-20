"""GPU-only checks invoked by amd_preflight, with imports deferred on the VM."""

import argparse
import importlib
import importlib.metadata
import json
import os
import random
import subprocess
import time
from pathlib import Path

from experiments.tcs_async_rl.amd_preflight import CONFIG, ROOT, load_configs


def check_imports(run_dir, config_path):
    # These optional dependencies exist only in the AMD container.
    import torch

    qualification, _ = load_configs(config_path)
    if not torch.version.hip or torch.cuda.device_count() != qualification["allocation"]["gpus"]:
        raise RuntimeError("Expected ROCm PyTorch and eight visible GPUs")
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        architecture = properties.gcnArchName
        if "gfx950" not in architecture or properties.total_memory < 256 * 2**30:
            raise RuntimeError("Expected MI350-class gfx950 devices with at least 256 GiB")
        devices.append(
            {"index": index, "name": properties.name, "architecture": architecture, "bytes": properties.total_memory}
        )
    modules = (
        "miles",
        "megatron.core",
        "sglang",
        "flash_attn",
        "fla",
        "transformer_engine",
        "apex",
        "miles_plugins.models.qwen3_5",
    )
    paths = {name: str(importlib.import_module(name).__file__) for name in modules}
    if not Path(paths["miles"]).resolve().is_relative_to(ROOT):
        raise RuntimeError("Container's baked Miles shadows the experiment checkout")
    versions = {}
    for name in ("torch", "sglang", "transformers", "flash-linear-attention", "transformer-engine", "ray"):
        versions[name] = importlib.metadata.version(name)
    commits = {}
    for name, path in (("megatron", "/root/Megatron-LM"), ("sglang", "/sgl-workspace/sglang")):
        commits[name] = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    evidence = {
        "devices": devices,
        "torch_hip": torch.version.hip,
        "versions": versions,
        "module_paths": paths,
        "dependency_commits": commits,
    }
    (run_dir / "environment.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2), flush=True)


def check_collective():
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        values = torch.ones(524288, device="cuda", dtype=torch.float32)
        dist.all_reduce(values)
        torch.cuda.synchronize()
        if not torch.all(values == dist.get_world_size()).item():
            raise RuntimeError("RCCL collective returned incorrect values")
        print(f"rank {dist.get_rank()}: 2 MiB all-reduce passed", flush=True)
    finally:
        dist.destroy_process_group()


def engine_arguments(model_dir, qualification, training):
    args = training["miles_args"]
    required = (
        "context_length",
        "mem_fraction_static",
        "max_running_requests",
    )
    optional = (
        "speculative_algorithm",
        "speculative_num_steps",
        "speculative_eagle_topk",
        "speculative_num_draft_tokens",
    )
    return {
        "model_path": str(model_dir),
        "tp_size": args["rollout_num_gpus_per_engine"],
        "random_seed": qualification["inference"]["seed"],
        "dtype": "bfloat16",
        "allow_auto_truncate": False,
        **{name: args[f"sglang_{name}"] for name in required},
        **{name: args[f"sglang_{name}"] for name in optional if f"sglang_{name}" in args},
    }


def validate_lengths(output, *, prompt_tokens, generated_tokens):
    metadata = output["meta_info"]
    if metadata["prompt_tokens"] != prompt_tokens:
        raise RuntimeError("SGLang changed the input length")
    if metadata["completion_tokens"] != generated_tokens:
        raise RuntimeError("SGLang did not reach the requested short decode length")


def check_inference(*, model_dir, config_path, run_dir):
    # SGLang owns the subprocess lifecycle, GPU workers, and shutdown.
    import sglang

    qualification, training = load_configs(config_path)
    arguments = engine_arguments(model_dir, qualification, training)
    (run_dir / "engine-arguments.json").write_text(json.dumps(arguments, indent=2) + "\n")
    engine = sglang.Engine(**arguments)
    try:
        results = []
        generated = qualification["inference"]["generated_tokens"]
        lengths = [qualification["inference"]["short_prompt_tokens"], arguments["context_length"] - generated]
        for length in lengths:
            rng = random.Random(qualification["inference"]["seed"] + length)
            tokens = [rng.randrange(1000, 20000) for _ in range(length)]
            (run_dir / f"synthetic-input-{length}.json").write_text(json.dumps(tokens) + "\n")
            started = time.monotonic()
            output = engine.generate(
                input_ids=tokens, sampling_params={"temperature": 0, "max_new_tokens": generated, "ignore_eos": True}
            )
            elapsed = time.monotonic() - started
            (run_dir / f"synthetic-output-{length}.json").write_text(json.dumps(output, indent=2) + "\n")
            validate_lengths(output, prompt_tokens=length, generated_tokens=generated)
            results.append({"prompt_tokens": length, "generated_tokens": generated, "elapsed_seconds": elapsed})
        evidence = {
            "cases": results,
            "scope": "Single-request prefill plus 32-token decode. Not 90k autoregressive generation or 64-way concurrency.",
        }
        (run_dir / "inference-results.json").write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps(evidence, indent=2), flush=True)
    finally:
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("imports", "collective", "inference"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    if args.mode == "imports":
        check_imports(args.run_dir, args.config)
    elif args.mode == "collective":
        check_collective()
    else:
        check_inference(model_dir=args.model_dir, config_path=args.config, run_dir=args.run_dir)


if __name__ == "__main__":
    main()
