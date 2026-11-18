"""Generate one real TCS response group, saving each completion independently."""

import argparse
import asyncio
import datetime
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

from experiments.tcs_async_rl.amd_preflight import load_configs
from experiments.tcs_async_rl.gpu_qualification import engine_arguments


def load_input(config, training):
    path = Path(config["input_file"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != config["input_sha256"]:
        raise ValueError("Private input differs from the approved diagnostic artifact")
    row = json.loads(path.read_text())
    if row["cohort"] not in training["data"]["eligible_configs"]:
        raise ValueError("Diagnostic input must belong to an eligible cohort")
    for key in ("cohort", "row", "paper_id", "node_id"):
        if row[key] != config["selection"][key]:
            raise ValueError(f"Input identity mismatch: {key}")
    return row


def final_answer(text):
    if "</think>" not in text:
        return None
    answer = text.rsplit("</think>", 1)[1].strip()
    for marker in ("<|im_end|>", "<|endoftext|>"):
        answer = answer.removesuffix(marker).strip()
    return answer


def _append_json(path, row):
    with path.open("a") as output:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())


async def generate_group(engine, *, token_ids, training, seed, run_dir):
    args = training["miles_args"]
    sampling = {
        "temperature": args["rollout_temperature"],
        "top_p": args["rollout_top_p"],
        "top_k": args["rollout_top_k"],
        "max_new_tokens": args["rollout_max_response_len"],
        # Keep delimiters in the raw diagnostic artifact for answer extraction.
        # The extracted final answer omits the terminal marker.
        "skip_special_tokens": False,
    }

    async def one(index):
        started = time.monotonic()
        _append_json(run_dir / "requests.jsonl", {"event": "started", "sample_index": index, "seed": seed + index})
        try:
            output = await engine.async_generate(
                input_ids=token_ids, sampling_params={**sampling, "sampling_seed": seed + index}
            )
            if output["meta_info"]["prompt_tokens"] != len(token_ids):
                raise RuntimeError("Engine changed the prompt length")
            record = {
                "sample_index": index,
                "seed": seed + index,
                "elapsed_seconds": time.monotonic() - started,
                "output": output,
                "final_answer": final_answer(output.get("text", "")),
            }
            _append_json(run_dir / "responses.jsonl", record)
            print(f"Completed sample {index}: {output['meta_info']['completion_tokens']} generated tokens", flush=True)
            return record
        except Exception as error:
            _append_json(
                run_dir / "requests.jsonl",
                {"event": "failed", "sample_index": index, "error_type": type(error).__name__},
            )
            raise

    return await asyncio.gather(*(one(i) for i in range(args["n_samples_per_prompt"])))


async def progress_monitor(run_dir, started, count):
    while True:
        path = run_dir / "responses.jsonl"
        completed = len(path.read_text().splitlines()) if path.exists() else 0
        print(f"Group progress: {completed}/{count} complete; elapsed {time.monotonic() - started:.0f}s", flush=True)
        await asyncio.sleep(30)


async def run_group(*, config_path, model_dir, run_dir):
    # GPU dependencies are deferred so the dataset boundary and parser are CPU-testable.
    import sglang
    import torch
    from transformers import AutoTokenizer

    config, training = load_configs(config_path)
    if not torch.version.hip or torch.cuda.device_count() != config["allocation"]["gpus"]:
        raise RuntimeError("Unexpected GPU visibility or non-ROCm PyTorch")
    row = load_input(config, training)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["problem"]}],
        tokenize=False,
        add_generation_prompt=True,
        **training["miles_args"]["apply_chat_template_kwargs"],
    )
    if formatted != row["formatted_prompt"]:
        raise ValueError("Container chat template differs from the validated input")
    token_ids = tokenizer.encode(formatted, add_special_tokens=False)
    if (
        len(token_ids) + training["miles_args"]["rollout_max_response_len"]
        > training["miles_args"]["sglang_context_length"]
    ):
        raise ValueError("Prompt plus output cap exceeds engine context")
    arguments = engine_arguments(model_dir, config, training)
    arguments.update(
        tp_size=config["inference"]["tensor_parallel_size"],
        max_running_requests=config["inference"]["max_running_requests"],
        skip_server_warmup=True,
    )
    (run_dir / "engine-arguments.json").write_text(json.dumps(arguments, indent=2) + "\n")
    (run_dir / "input.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
    (run_dir / "formatted-prompt.txt").write_text(formatted)
    properties = torch.cuda.get_device_properties(0)
    environment = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "sglang": importlib.metadata.version("sglang"),
        "transformers": importlib.metadata.version("transformers"),
        "gpu": properties.name,
        "gpu_memory_bytes": properties.total_memory,
        "prompt_tokens": len(token_ids),
    }
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(json.dumps(environment), flush=True)
    engine = sglang.Engine(**arguments)
    started = time.monotonic()
    monitor = asyncio.create_task(progress_monitor(run_dir, started, training["miles_args"]["n_samples_per_prompt"]))
    try:
        responses = await generate_group(
            engine, token_ids=token_ids, training=training, seed=config["inference"]["seed"], run_dir=run_dir
        )
        summary = {
            "completed": len(responses),
            "with_final_answer": sum(bool(r["final_answer"]) for r in responses),
            "generated_tokens": [r["output"]["meta_info"]["completion_tokens"] for r in responses],
            "elapsed_seconds": time.monotonic() - started,
            "finished_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "judge_status": "not_graded",
        }
        (run_dir / "generation-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    finally:
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run_group(config_path=args.config, model_dir=args.model_dir, run_dir=args.run_dir))


if __name__ == "__main__":
    main()
