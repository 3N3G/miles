"""Run the approved two-update runtime check inside an isolated AMD allocation.

The host preflight owns Slurm/container preparation. Existing command_utils owns
checkpoint conversion and Ray. This module never transfers API credentials.
"""

import argparse
import hashlib
import json
import os
import shlex
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from experiments.tcs_async_rl.amd_preflight import load_configs
from experiments.tcs_async_rl.group_inference import load_input
from experiments.tcs_async_rl.reward_hook import prepare_smoke_row, question_key
from experiments.tcs_async_rl.rl_artifacts import export_rollouts


def validate_group_gate(grades, expected_samples):
    scores = grades["scores_by_case"]
    candidates = [scores[key] for key in sorted(scores) if key.startswith("sample-")]
    if grades["scored_responses"] != expected_samples or len(candidates) != expected_samples:
        raise ValueError("The complete diagnostic response group must be graded first")
    if grades["incomplete_or_empty_final_answers"] or len(set(candidates)) < 2:
        raise ValueError("Diagnostic requires completed answers and non-constant real rubric rewards")
    if [scores.get(key) for key in ("control-reference", "control-empty", "control-irrelevant")] != [7, 0, 0]:
        raise ValueError("Rubric controls did not pass")
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 7 for value in candidates
    ):
        raise ValueError("Diagnostic grades are outside the supplied rubric")


def resolved_args(config, training, *, work_dir, run_dir, run_id):
    args = {**training["miles_args"], **config["overrides"]}
    model = training["model"]
    model_name = model["repo_id"].split("/")[-1] + "-" + model["revision"]
    hf = work_dir / "models" / model_name
    conversion_name = (
        model_name
        + "_"
        + "_".join(
            f"{short}{args[key]}"
            for short, key in (
                ("mtp", "mtp_num_layers"),
                ("tp", "tensor_model_parallel_size"),
                ("pp", "pipeline_model_parallel_size"),
                ("cp", "context_parallel_size"),
            )
        )
    )
    args.update(
        hf_checkpoint=str(hf),
        ref_load=str(work_dir / "models" / (conversion_name + "_torch_dist")),
        load=str(run_dir / "checkpoints"),
        save=str(run_dir / "checkpoints"),
        prompt_data=str(run_dir / "input.jsonl"),
        wandb_dir=str(run_dir / "wandb"),
        wandb_group=run_id,
        wandb_run_id=run_id,
        tb_project_name=str(run_dir / "tensorboard"),
        tb_experiment_name=run_id,
        save_debug_rollout_data=str(run_dir / "rollout_data" / "rollout-{rollout_id}.pt"),
    )
    if any(value is None for value in args.values()):
        raise ValueError("Every Miles argument must be resolved before submission")
    return args, conversion_name


def native_argv(args):
    argv = []
    for key, value in args.items():
        if value is False:
            continue
        argv.append("--" + key.replace("_", "-"))
        if value is not True:
            argv.append(json.dumps(value, separators=(",", ":")) if isinstance(value, dict) else str(value))
    return argv


def wait_for_judge(url, expected_key, *, judge, seconds=600):
    health = url.rsplit("/", 1)[0] + "/health"
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=5) as response:
                status = json.load(response)
            if status.get("question_key") != expected_key or status.get("ready") is not True:
                raise ValueError("Reward service does not match the approved question")
            if (
                status.get("model") != judge["primary_model"]
                or status.get("reasoning_effort") != judge["reasoning_effort"]
            ):
                raise ValueError("Reward service changed the approved judge or effort")
            print("VM reward tunnel ready", flush=True)
            return
        except (urllib.error.URLError, TimeoutError):
            print("Waiting for VM reward tunnel", flush=True)
            time.sleep(10)
    raise RuntimeError("Reward tunnel was not available within ten minutes")


def main():
    # GPU/runtime dependencies stay out of the CPU configuration and gate checks.
    import torch
    from transformers import AutoTokenizer

    from miles.utils.external_utils import command_utils as U
    from miles.utils.external_utils.command_utils.common import train_env_vars

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--validate-arguments-only", action="store_true")
    cli = parser.parse_args()
    config, training = load_configs(cli.config)
    if not os.environ.get("SLURM_JOB_ID") or not torch.version.hip:
        raise RuntimeError("Requires a ROCm Slurm allocation")
    # Broad upstream launcher cleanup is safe only inside our private PID namespace.
    host_pid_namespace = os.environ.get("TCS_HOST_PID_NAMESPACE")
    if not host_pid_namespace or os.readlink("/proc/self/ns/pid") == host_pid_namespace:
        raise RuntimeError("Host preflight must verify a private PID namespace before Ray cleanup")
    # Conversion precedes Ray submission. Reuse its Megatron communication defaults here too.
    os.environ.update(
        train_env_vars(
            SimpleNamespace(train_backend_fsdp=False, extra_env_vars={}),
            {},
            config=U.ExecuteTrainConfig(output_dir=str(cli.run_dir)),
        )
    )
    if cli.validate_arguments_only:
        from miles.utils.arguments import parse_args
        from miles.utils.external_utils.model_args_utils import load_model_args

        args, _ = resolved_args(
            config, training, work_dir=cli.work_dir, run_dir=cli.run_dir, run_id="tcs-argument-check"
        )
        sys.argv = [sys.argv[0], *shlex.split(load_model_args("qwen3.5-4B")), *native_argv(args)]
        parsed = parse_args()
        (cli.run_dir / "validated-native-arguments.json").write_text(
            json.dumps({"requested": args, "validated": vars(parsed), "optimizer_updates": 0}, indent=2, default=str)
            + "\n"
        )
        print("Native argument validation and private PID namespace passed. No launcher started.", flush=True)
        return
    if torch.cuda.device_count() != config["allocation"]["gpus"]:
        raise RuntimeError("Requires the approved GPU count before RL execution")
    input_config, _ = load_configs(cli.config.parent / config["input_config"])
    row = load_input(input_config, training)
    gate_file = Path(config["gate_file"])
    grades = json.loads(gate_file.read_text())
    if grades.get("input_sha256") != input_config["input_sha256"]:
        raise ValueError("Group gate is not tied to the approved private input hash")
    validate_group_gate(grades, training["miles_args"]["n_samples_per_prompt"])
    run_id = U.create_run_id()
    args, conversion_name = resolved_args(config, training, work_dir=cli.work_dir, run_dir=cli.run_dir, run_id=run_id)
    tokenizer = AutoTokenizer.from_pretrained(args["hf_checkpoint"], local_files_only=True)
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["problem"]}],
        tokenize=False,
        add_generation_prompt=True,
        **args["apply_chat_template_kwargs"],
    )
    if formatted != row["formatted_prompt"]:
        raise ValueError("Training prompt differs from the approved group diagnostic")
    (cli.run_dir / "input.jsonl").write_text(json.dumps(prepare_smoke_row(row), ensure_ascii=False) + "\n")
    (cli.run_dir / "formatted-prompt.txt").write_text(formatted)
    (cli.run_dir / "rollout_data").mkdir(exist_ok=True)
    (cli.run_dir / "resolved-config.json").write_text(
        json.dumps(
            {
                "args": args,
                "run_id": run_id,
                "gate_sha256": hashlib.sha256(gate_file.read_bytes()).hexdigest(),
                "question_key": question_key(row),
            },
            indent=2,
        )
        + "\n"
    )
    wait_for_judge(args["rm_url"], question_key(row), judge=training["judge"])
    os.environ.update(
        NCCL_NVLS_ENABLE="0",
        RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES="1",
        RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1",
        # The native TensorBoard adapter otherwise prefixes even absolute paths.
        TENSORBOARD_DIR=str(cli.run_dir / "tensorboard"),
        WANDB_CACHE_DIR=str(cli.work_dir / "cache" / "wandb"),
        WANDB_CONFIG_DIR=str(cli.run_dir / "wandb" / "config"),
        WANDB_DATA_DIR=str(cli.run_dir / "wandb" / "data"),
    )
    if devices := os.environ.get("HIP_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
    backend = U.ExecuteTrainConfig(output_dir=str(cli.run_dir), run_id=run_id).create_backend()
    conversion_args = {
        key: args[key]
        for key in (
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
            "mtp_num_layers",
            "bf16",
            "linear_attention_backend",
            "attention_backend",
        )
    }
    backend.convert_checkpoint(
        model_name=conversion_name,
        megatron_model_type="qwen3.5-4B",
        num_gpus_per_node=args["actor_num_gpus_per_node"],
        dir_dst=str(cli.work_dir / "models"),
        hf_checkpoint=args["hf_checkpoint"],
        extra_args=shlex.join(native_argv(conversion_args)),
    )
    caches = {
        key: os.environ[key]
        for key in (
            "AITER_JIT_DIR",
            "AITER_ROOT_DIR",
            "SGLANG_CACHE_DIR",
            "XDG_CACHE_HOME",
            "TRITON_CACHE_DIR",
            "TORCH_EXTENSIONS_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "HF_HOME",
            "MILES_HARDWARE_PLATFORM",
            "RAY_TMPDIR",
            "TENSORBOARD_DIR",
            "WANDB_CACHE_DIR",
            "WANDB_CONFIG_DIR",
            "WANDB_DATA_DIR",
        )
    }
    try:
        backend.execute_train(
            train_args=shlex.join(native_argv(args)),
            num_gpus_per_node=args["num_gpus_per_node"],
            megatron_model_type="qwen3.5-4B",
            train_script="train_async.py",
            extra_env_vars=caches,
            job_lifetime="launcher",
        )
    finally:
        export_rollouts(cli.run_dir)
    (cli.run_dir / "training-finished.json").write_text(
        json.dumps({"launcher_completed": True, "run_id": run_id}) + "\n"
    )


if __name__ == "__main__":
    main()
