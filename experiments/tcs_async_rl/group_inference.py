"""Generate one real TCS response group, saving each completion independently."""

import argparse
import asyncio
import datetime
import hashlib
import importlib.metadata
import json
import math
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


def sampling_parameters(training):
    args = training["miles_args"]
    return {
        "temperature": args["rollout_temperature"],
        "top_p": args["rollout_top_p"],
        "top_k": args["rollout_top_k"],
        "max_new_tokens": args["rollout_max_response_len"],
        # Keep delimiters in the raw diagnostic artifact for answer extraction.
        # The extracted final answer omits the terminal marker.
        "skip_special_tokens": False,
    }


def validate_sampling_support_control(output, *, expected_tokens, top_k):
    token_ids = output["output_ids"]
    supports = output["meta_info"]["output_token_sampling_mask"]
    logprobs = output["meta_info"]["output_token_sampling_logprobs"]
    if len(token_ids) != expected_tokens or len(supports) != expected_tokens or len(logprobs) != expected_tokens:
        raise ValueError("Sampling support control has incomplete token metadata")
    for token, support, probabilities in zip(token_ids, supports, logprobs, strict=True):
        if not 0 < len(support) <= top_k or len(set(support)) != len(support) or token not in support:
            raise ValueError("Sampling support does not contain the sampled token within its declared bound")
        if len(probabilities) != len(support) or not all(math.isfinite(p) for p in probabilities):
            raise ValueError("Sampling support control has invalid log probabilities")
        if abs(sum(math.exp(p) for p in probabilities) - 1) > 1e-4:
            raise ValueError("Sampling support log probabilities are not normalized")


async def check_sampling_support(engine, *, token_ids, training, seed, run_dir):
    # The Engine convenience method omits these native request fields in this image.
    # Use the same request schema and tokenizer-manager owner as its HTTP endpoint.
    from sglang.srt.managers.io_struct import GenerateReqInput

    params = {**sampling_parameters(training), "max_new_tokens": 16, "sampling_seed": seed}
    request = GenerateReqInput(
        input_ids=token_ids,
        sampling_params=params,
        return_logprob=True,
        return_sampling_mask=True,
        sampling_logprobs_mode="support",
    )
    output = await engine.tokenizer_manager.generate_request(request, None).__anext__()
    (run_dir / "sampling-support-control.json").write_text(json.dumps(output, indent=2) + "\n")
    validate_sampling_support_control(output, expected_tokens=16, top_k=params["top_k"])
    print("Native bounded-sampling support and normalized behavior log probabilities passed", flush=True)


async def generate_group(engine, *, token_ids, training, seed, run_dir, validate_first_sample=None):
    args = training["miles_args"]
    sampling = sampling_parameters(training)

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

    indices = list(range(args["n_samples_per_prompt"]))
    validated = []
    if validate_first_sample is not None:
        if type(validate_first_sample) is not int or validate_first_sample not in indices:
            raise ValueError("First validation sample must belong to the declared group")
        record = await one(validate_first_sample)
        if not record["final_answer"] or record["output"]["meta_info"]["finish_reason"]["type"] != "stop":
            raise RuntimeError("Previously failed seed did not complete under the proposed sampling repair")
        validated.append(record)
        indices.remove(validate_first_sample)
        print("Previously failed seed completed. Starting the remaining group.", flush=True)
    return validated + await asyncio.gather(*(one(i) for i in indices))


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
    (run_dir / "sampling-parameters.json").write_text(json.dumps(sampling_parameters(training), indent=2) + "\n")
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
        if config["inference"].get("validate_sampling_support", False):
            await check_sampling_support(
                engine, token_ids=token_ids, training=training, seed=config["inference"]["seed"], run_dir=run_dir
            )
        responses = await generate_group(
            engine,
            token_ids=token_ids,
            training=training,
            seed=config["inference"]["seed"],
            run_dir=run_dir,
            validate_first_sample=config["inference"].get("validate_first_sample"),
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
