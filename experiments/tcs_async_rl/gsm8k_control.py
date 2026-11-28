"""Run one approved GSM8K arm in a private AMD container.

The approved configuration owns scientific settings. The native command backend
owns Ray and training. This wrapper supplies bounded Slurm execution and evidence.
"""

import argparse
import datetime
import importlib.metadata
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
from pathlib import Path

from experiments.tcs_async_rl.amd_preflight import _public_environment, _run, container_prefix
from experiments.tcs_async_rl.gsm8k_artifacts import write_json
from experiments.tcs_async_rl.rl_artifacts import file_sha256
from experiments.tcs_async_rl.rl_smoke import native_argv

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("gsm8k_control.approved_20261126.json")
MODEL_TYPE = "qwen2.5-0.5B"


def configure_bridge_cache(work_dir):
    directory = work_dir / "cache" / "megatron-config-locks"
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["MEGATRON_CONFIG_LOCK_DIR"] = str(directory)
    return directory


def forwarded_runtime_environment(environment):
    allowed = {"HF_HOME", "XDG_CACHE_HOME", "RAY_TMPDIR", "MEGATRON_CONFIG_LOCK_DIR",
               "MILES_HARDWARE_PLATFORM", "AITER_JIT_DIR", "AITER_ROOT_DIR", "MIOPEN_USER_DB_PATH",
               "MIOPEN_CUSTOM_CACHE_DIR", "NCCL_NVLS_ENABLE", "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
               "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"}
    return {key: environment[key] for key in environment if key.endswith("CACHE_DIR") or key in allowed}


def resolved_args(config, *, arm, work_dir, run_dir, run_id):
    if config["status"] != "approved_20261126" or arm not in ("reference", "cispo"):
        raise ValueError("Requires the approved control and a declared arm")
    shared, allocation = config["shared"], config["allocation"]
    selected = next(item for item in config["arms"] if item["id"] == arm)
    model = config["model"]
    hf = work_dir / "models" / (model["repo_id"].split("/")[-1] + "-" + model["revision"])
    excluded = {"entrypoint", "maximum_optimizer_updates", "seeds"}
    args = {key: value for key, value in shared.items() if key not in excluded}
    # The native parser derives support replay from top-p/top-k; it has no CLI switch.
    args.update({key: value for key, value in selected.items()
                 if key not in {"id", "source_recipe", "role", "use_sampling_support_replay"}})
    args.update(
        num_rollout=shared["maximum_optimizer_updates"], seed=shared["seeds"][0],
        hf_checkpoint=str(hf), ref_load=str(hf), load=str(hf), finetune=True,
        no_load_optim=True, no_load_rng=True, save=str(run_dir / "checkpoints"), save_interval=20,
        ckpt_format="torch_dist", megatron_to_hf_mode="bridge", bf16=True,
        prompt_data=str(run_dir / "train.jsonl"),
        eval_prompt_data=["gsm8k", str(run_dir / "validation.jsonl")],
        actor_num_nodes=1, num_gpus_per_node=allocation["physical_gpus"],
        actor_num_gpus_per_node=allocation["actor_gpus"], rollout_num_gpus=allocation["rollout_gpus"],
        rollout_num_gpus_per_engine=allocation["rollout_gpus_per_engine"],
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1,
        expert_model_parallel_size=1, expert_tensor_parallel_size=1, sequence_parallel=True,
        use_dynamic_batch_size=True, attention_backend="flash", attention_softmax_in_fp32=True,
        attention_dropout=0.0, hidden_dropout=0.0, accumulate_allreduce_grads_in_fp32=True,
        advantage_estimator="grpo", use_kl_loss=True, kl_loss_type="low_var_kl",
        sglang_mem_fraction_static=0.7, sglang_enable_metrics=True,
        seq_length=4096, max_position_embeddings=32768, sglang_context_length=4096,
        use_wandb=False, wandb_team="geney", wandb_project="tcs-async-rl",
        wandb_run_id=run_id, wandb_group="gsm8k-control", wandb_always_use_train_step=True,
        observe_training_entropy=True, observe_update_direction_steps=[0, 1, 10],
        check_weight_update_equal=True, log_inference_engine_weight_checksums=True,
        save_debug_event_data=str(run_dir / "events"),
        save_debug_rollout_data=str(run_dir / "rollout_data" / "rollout-{rollout_id}.pt"),
        custom_rollout_log_function_path="experiments.tcs_async_rl.gsm8k_artifacts.log_training",
        custom_megatron_post_save_hook_path="experiments.tcs_async_rl.rl_artifacts.retain_latest_checkpoint",
        save_trigger_sentinel=str(run_dir / "checkpoint-request"), stop_after_save_trigger=True,
    )
    return args


def validate_derived_arguments(config, arm, parsed):
    selected = next(item for item in config["arms"] if item["id"] == arm)
    if "use_sampling_support_replay" in selected:
        if parsed.use_sampling_support_replay != selected["use_sampling_support_replay"]:
            raise ValueError("Native sampling-support replay differs from the approved setting")


def prepare_inputs(source_dir, run_dir, preparation):
    expected = preparation["partition_sha256"]
    for name, count in (("train", 7217), ("validation", 256)):
        source = source_dir / f"{name}.jsonl"
        if file_sha256(source) != expected[name]:
            raise ValueError("Prepared GSM8K split changed")
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        if len(rows) != count:
            raise ValueError("Wrong GSM8K partition size")
        with (run_dir / f"{name}.jsonl").open("w") as stream:
            for row in rows:
                row["metadata"] = {"source_row": row["source_row"], "control_split": name}
                stream.write(json.dumps(row, ensure_ascii=True) + "\n")
    write_json(run_dir / "input-provenance.json", preparation)


def render_job(*, repo, work_dir, run_dir, config, config_path, arm, expected_commit, seconds=7200,
               cpus_per_task=64):
    if arm not in ("reference", "cispo") or not 60 <= seconds <= 7200:
        raise ValueError("An attempt must fit the approved arm and remaining two-hour budget")
    minutes = seconds // 60
    walltime = f"{minutes // 60:02d}:{minutes % 60:02d}:00"
    allocation = config["allocation"]
    argv = ["/usr/bin/python3.12", "-m", "experiments.tcs_async_rl.gsm8k_control", "--phase", "host",
            "--config", str(config_path), "--arm", arm, "--work-dir", str(work_dir),
            "--run-dir", str(run_dir), "--expected-commit", expected_commit]
    return "\n".join([
        "#!/usr/bin/python3.12", f"#SBATCH --job-name=gsm8k-{arm}",
        f"#SBATCH --partition={allocation['partition']}", f"#SBATCH --account={allocation['account']}",
        f"#SBATCH --qos={allocation['qos']}", "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus_per_task}", "#SBATCH --exclusive", "#SBATCH --no-requeue",
        f"#SBATCH --time={walltime}",
        f"#SBATCH --chdir={repo}", f"#SBATCH --output={run_dir}/slurm-%j.log",
        "import subprocess", f"subprocess.run({argv!r}, check=True)", "",
    ])


def run_host(cli, config):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_PARTITION") != config["allocation"]["partition"]:
        raise RuntimeError("Requires this control's approved AMD allocation")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    if commit != cli.expected_commit or dirty:
        raise RuntimeError("Allocation checkout differs from its frozen submission")
    if (cli.run_dir / "manifest.json").exists():
        raise ValueError("Use a fresh run directory")
    qualification = json.loads((cli.config.parent / config["allocation"]["container_recipe"]).read_text())
    image = cli.work_dir / "containers" / (qualification["container"]["uri"].split("sha256:")[1] + ".sandbox")
    if not (image / ".miles-ready").is_file() or shutil.disk_usage(cli.work_dir).free < 40 * 2**30:
        raise RuntimeError("Container cache or storage preflight failed")
    write_json(cli.run_dir / "manifest.json", {
        "job_id": os.environ["SLURM_JOB_ID"], "commit": commit,
        "started_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "training_proposal": config, "qualification": qualification, "arm": cli.arm,
    })
    runtime = cli.run_dir / "ray"
    runtime.mkdir(exist_ok=True)
    prefix = container_prefix(image=image, repo=ROOT, work_dir=cli.work_dir,
                              isolate_processes=True, runtime_dir=runtime,
                              cache_namespace=config["allocation"]["partition"])
    # Leave fifteen minutes for a completed save, final eval and orderly cleanup.
    def request_save():
        if not (cli.run_dir / "training-finished.json").exists():
            (cli.run_dir / "checkpoint-request").touch()
    seconds = json.loads((cli.run_dir / "submission.json").read_text())["walltime_seconds"]
    if not 60 <= seconds <= 7200:
        raise ValueError("Invalid allocation budget")
    timer = threading.Timer(max(1, seconds - 15 * 60), request_save)
    timer.daemon = True
    timer.start()
    try:
        _run("gsm8k-training", prefix + ["python3", "-m", "experiments.tcs_async_rl.gsm8k_control",
             "--phase", "gpu", "--config", str(cli.config), "--arm", cli.arm,
             "--work-dir", str(cli.work_dir), "--run-dir", str(cli.run_dir)],
             run_dir=cli.run_dir, env=_public_environment())
    finally:
        timer.cancel()


def run_gpu(cli, config):
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    bridge_cache = configure_bridge_cache(cli.work_dir)
    import torch
    from megatron.bridge import AutoBridge
    from transformers import AutoTokenizer
    from miles.utils.arguments import parse_args
    from miles.utils.external_utils import command_utils as U
    from miles.utils.external_utils.model_args_utils import load_model_args

    namespace = os.environ.get("TCS_HOST_PID_NAMESPACE")
    physical_gpus = config["allocation"]["physical_gpus"]
    if not torch.version.hip or torch.cuda.device_count() != physical_gpus or not namespace:
        raise RuntimeError(f"Requires exactly {physical_gpus} allocated ROCm GPUs")
    if os.readlink("/proc/self/ns/pid") == namespace:
        raise RuntimeError("Ray cleanup requires our private PID namespace")
    run_id = f"gsm8k-{cli.arm}-" + U.create_run_id()
    args = resolved_args(config, arm=cli.arm, work_dir=cli.work_dir, run_dir=cli.run_dir, run_id=run_id)
    backend = U.ExecuteTrainConfig(output_dir=str(cli.run_dir), run_id=run_id).create_backend()
    backend.exec_command_cpu(shlex.join(["hf", "download", config["model"]["repo_id"], "--revision",
                            config["model"]["revision"], "--local-dir", args["hf_checkpoint"]]))
    tokenizer = AutoTokenizer.from_pretrained(args["hf_checkpoint"], local_files_only=True)
    example = json.loads(next((cli.run_dir / "train.jsonl").open()))
    prompt = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
    if prompt != (cli.run_dir / "expected-prompt.txt").read_text():
        raise ValueError("Allocation tokenizer differs from the prepared pinned example")
    (cli.run_dir / "formatted-prompt.txt").write_text(prompt)
    sys.argv = [sys.argv[0], *shlex.split(load_model_args(MODEL_TYPE)), *native_argv(args)]
    parsed = parse_args()
    validate_derived_arguments(config, cli.arm, parsed)
    write_json(cli.run_dir / "validated-native-arguments.json",
               {"requested": args, "validated": vars(parsed)}, default=str)
    # Check the actor's real config loader before starting six rollout engines.
    bridge = AutoBridge.from_hf_pretrained(args["hf_checkpoint"], trust_remote_code=True)
    write_json(cli.run_dir / "bridge-cache-preflight.json", {
        "directory": str(bridge_cache), "loader": "AutoBridge.from_hf_pretrained",
        "loaded_type": type(bridge).__name__, "passed": True,
    })
    write_json(cli.run_dir / "resolved-config.json", {
        "args": args, "run_id": run_id, "benchmark": "gsm8k", "arm": cli.arm,
        "dataset_sha256": {name: file_sha256(cli.run_dir / f"{name}.jsonl") for name in ("train", "validation")},
        "runtime_versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "ray")},
    })
    os.environ.update(NCCL_NVLS_ENABLE="0", RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES="1",
                      RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1")
    if os.environ.get("HIP_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["HIP_VISIBLE_DEVICES"]
    forwarded = forwarded_runtime_environment(os.environ)
    backend.execute_train(train_args=shlex.join(native_argv(args)), num_gpus_per_node=physical_gpus,
                          megatron_model_type=MODEL_TYPE, train_script="experiments/tcs_async_rl/gsm8k_driver.py",
                          extra_env_vars=forwarded, job_lifetime="launcher")
    write_json(cli.run_dir / "training-finished.json", {"launcher_completed": True, "run_id": run_id})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("host", "gpu"), required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--arm", choices=("reference", "cispo"), required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-commit")
    cli = parser.parse_args()
    config = json.loads(cli.config.read_text())
    if config["status"] != "approved_20261126":
        raise ValueError("Unapproved control")
    (run_host if cli.phase == "host" else run_gpu)(cli, config)


if __name__ == "__main__":
    main()
