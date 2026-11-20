"""Fixed-policy diagnostic using the existing compatible AMD vLLM image.

This is a generation/reward check. It does not qualify Miles' SGLang trainer or
claim performance equivalence between serving implementations.
"""

import argparse
import asyncio
import datetime
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from experiments.tcs_async_rl.amd_preflight import load_configs
from experiments.tcs_async_rl.group_inference import generate_group, load_input, progress_monitor, sampling_parameters


def completion_payload(token_ids, sampling, model):
    return {
        "model": model,
        "prompt": token_ids,
        "max_tokens": sampling["max_new_tokens"],
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "top_k": sampling["top_k"],
        "seed": sampling["sampling_seed"],
        "skip_special_tokens": False,
        "add_special_tokens": False,
        "n": 1,
        "stream": False,
    }


def completion_output(reply):
    if len(reply["choices"]) != 1:
        raise ValueError("Expected one completion per independently seeded request")
    choice = reply["choices"][0]
    if choice["finish_reason"] not in ("stop", "length"):
        raise ValueError("Generator did not finish normally")
    return {
        "text": choice["text"],
        "meta_info": {
            "prompt_tokens": reply["usage"]["prompt_tokens"],
            "completion_tokens": reply["usage"]["completion_tokens"],
            "finish_reason": {"type": choice["finish_reason"]},
        },
        "raw_api_response": reply,
    }


class CompletionClient:
    def __init__(self, session, url, model):
        self.session, self.url, self.model = session, url, model

    async def async_generate(self, *, input_ids, sampling_params):
        async with self.session.post(
            self.url + "/v1/completions", json=completion_payload(input_ids, sampling_params, self.model)
        ) as response:
            response.raise_for_status()
            return completion_output(await response.json())


async def wait_ready(session, url, process, *, seconds=1200):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("vLLM server exited during startup; inspect vllm-server.log")
        try:
            async with session.get(url + "/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, TimeoutError):
            pass
        await asyncio.sleep(5)
    raise RuntimeError("vLLM server did not become ready within twenty minutes")


async def run(*, config_path, model_dir, run_dir):
    # The allocation image supplies these packages. VM parsing/tests stay GPU-free.
    import aiohttp
    import torch
    from transformers import AutoTokenizer

    config, training = load_configs(config_path)
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    if config["inference"]["speculative_decoding"] is not False or config["inference"]["enforce_eager"] is not True:
        raise ValueError("This qualified serving repair requires eager, target-model decoding")
    if not os.environ.get("SLURM_JOB_ID") or not torch.version.hip or torch.cuda.device_count() != 1:
        raise RuntimeError("Requires the approved one-GPU ROCm allocation")
    row = load_input(config, training)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    args = training["miles_args"]
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["problem"]}],
        tokenize=False,
        add_generation_prompt=True,
        **args["apply_chat_template_kwargs"],
    )
    if formatted != row["formatted_prompt"]:
        raise ValueError("Serving template differs from the approved thinking prompt")
    token_ids = tokenizer.encode(formatted, add_special_tokens=False)
    if len(token_ids) + args["rollout_max_response_len"] > args["rollout_max_context_len"]:
        raise ValueError("Prompt and output budget exceed the approved context")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    model_name = "tcs-qwen35-4b"
    argv = [
        "vllm",
        "serve",
        str(model_dir),
        "--served-model-name",
        model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        str(config["inference"]["tensor_parallel_size"]),
        "--max-model-len",
        str(args["rollout_max_context_len"]),
        "--language-model-only",
        "--enforce-eager",
        "--gpu-memory-utilization",
        str(args["sglang_mem_fraction_static"]),
        "--max-num-seqs",
        str(config["inference"]["max_running_requests"]),
        "--max-num-batched-tokens",
        "8192",
        "--no-enable-prefix-caching",
        "--generation-config",
        "vllm",
        "--seed",
        str(config["inference"]["seed"]),
    ]
    properties = torch.cuda.get_device_properties(0)
    if "min_gpu_memory_bytes" in config["inference"]:
        if properties.total_memory < config["inference"]["min_gpu_memory_bytes"]:
            raise RuntimeError("GPU has less memory than the configured concurrency requires")
    environment = {
        "backend": "vllm",
        "vllm": importlib.metadata.version("vllm"),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "transformers": importlib.metadata.version("transformers"),
        "gpu": properties.name,
        "architecture": properties.gcnArchName,
        "gpu_memory_bytes": properties.total_memory,
        "prompt_tokens": len(token_ids),
        "speculative_decoding": False,
        "scope": "Frozen-model generation/reward feasibility; Miles/SGLang runtime unqualified",
    }
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    (run_dir / "server-command.json").write_text(json.dumps(argv, indent=2) + "\n")
    (run_dir / "sampling-parameters.json").write_text(json.dumps(sampling_parameters(training), indent=2) + "\n")
    (run_dir / "input.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
    (run_dir / "formatted-prompt.txt").write_text(formatted)
    print(json.dumps(environment), flush=True)
    with (run_dir / "vllm-server.log").open("w") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        monitor = None
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=4 * 3600)) as session:
                url = f"http://127.0.0.1:{port}"
                await wait_ready(session, url, process)
                started = time.monotonic()
                monitor = asyncio.create_task(progress_monitor(run_dir, started, args["n_samples_per_prompt"]))
                responses = await generate_group(
                    CompletionClient(session, url, model_name),
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
                    "backend": "vllm",
                }
                (run_dir / "generation-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        finally:
            if monitor is not None:
                monitor.cancel()
                await asyncio.gather(monitor, return_exceptions=True)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(config_path=args.config, model_dir=args.model_dir, run_dir=args.run_dir))


if __name__ == "__main__":
    main()
