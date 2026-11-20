"""Render an AMD Slurm job, or execute it inside its allocation.

This job uses public model weights and synthetic token IDs. It makes no judge
calls and reads no private dataset. Rendering never submits a job.

    python3.12 -m experiments.tcs_async_rl.amd_preflight \
        --work-dir /work1/aviralkumar/geney/tcs-async-rl \
        --run-dir /work1/aviralkumar/geney/tcs-async-rl/runs/qualification-v1

Review the rendered Python batch file before passing it to sbatch. The caller
must create run-dir before submission because Slurm opens its output file first.
"""

import argparse
import datetime
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from experiments.tcs_async_rl.resource_estimate import model_inputs

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("qualification.proposed.json")
RAY_RUNTIME_PATH = "/tcs-ray"


def load_configs(path=CONFIG):
    qualification = json.loads(path.read_text())
    if "container_config" in qualification:
        qualification["container"] = json.loads((path.parent / qualification["container_config"]).read_text())[
            "container"
        ]
    training = json.loads((path.parent / qualification["training_config"]).read_text())
    return qualification, training


def render_job(*, repo, work_dir, run_dir, expected_commit, config_path=CONFIG):
    qualification, _ = load_configs(config_path)
    allocation = qualification["allocation"]
    for path in (repo, work_dir, run_dir, config_path):
        if any(char.isspace() for char in str(path)):
            raise ValueError("Slurm paths must not contain whitespace")
    argv = [
        qualification["host_python"],
        "-m",
        "experiments.tcs_async_rl.amd_preflight",
        "--config",
        str(config_path),
        "--work-dir",
        str(work_dir),
        "--run-dir",
        str(run_dir),
        "--expected-commit",
        expected_commit,
        "--run-in-allocation",
    ]
    return "\n".join(
        [
            f"#!{qualification['host_python']}",
            f"#SBATCH --job-name={qualification.get('job_name', 'tcs-amd-qualification')}",
            f"#SBATCH --partition={allocation['partition']}",
            f"#SBATCH --account={allocation['account']}",
            f"#SBATCH --qos={allocation['qos']}",
            f"#SBATCH --nodes={allocation['nodes']}",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={allocation.get('cpus_per_task', 1)}",
            "#SBATCH --exclusive",
            f"#SBATCH --time={allocation['walltime']}",
            f"#SBATCH --chdir={repo}",
            f"#SBATCH --output={run_dir}/slurm-%j.log",
            "import subprocess",
            f"subprocess.run({argv!r}, check=True)",
            "",
        ]
    )


def container_prefix(*, image, repo, work_dir, isolate_processes=False, runtime_dir=None, cache_namespace=None):
    cache = work_dir / "cache"
    gpu_cache = Path(image.name) / cache_namespace if cache_namespace else Path(image.name)
    environment = {
        "PYTHONPATH": f"{repo}:/root/Megatron-LM:/sgl-workspace/sglang/python",
        "MILES_HARDWARE_PLATFORM": "rocm",
        "HF_HOME": str(cache / "huggingface"),
        "XDG_CACHE_HOME": str(cache),
        "TRITON_CACHE_DIR": str(cache / "triton" / gpu_cache),
        "AITER_JIT_DIR": str(cache / "aiter" / gpu_cache),
        "AITER_ROOT_DIR": str(cache / "aiter_cpp" / gpu_cache),
        "SGLANG_CACHE_DIR": str(cache / "sglang" / gpu_cache),
        "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions" / gpu_cache),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "torch_inductor" / gpu_cache),
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for key in (
        "SLURM_JOB_ID",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_NUM_NODES",
        "SLURM_JOB_NODELIST",
        "SLURM_CPUS_PER_TASK",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ):
        if key in os.environ:
            environment[key] = os.environ[key]
    argv = [
        "apptainer",
        "exec",
        "--rocm",
        "--cleanenv",
        "--no-home",
        "--bind",
        f"{work_dir}:{work_dir}",
        "--bind",
        f"{repo}:{repo}",
        "--pwd",
        str(repo),
    ]
    if isolate_processes:
        if runtime_dir is None:
            raise ValueError("Isolated RL requires a run-owned Ray runtime directory")
        argv.append("--pid")
        argv += ["--bind", f"{runtime_dir}:{RAY_RUNTIME_PATH}"]
        environment["RAY_TMPDIR"] = RAY_RUNTIME_PATH
        environment["TCS_HOST_PID_NAMESPACE"] = os.readlink("/proc/self/ns/pid")
    for key, value in environment.items():
        argv += ["--env", f"{key}={value}"]
    return argv + [str(image)]


def _public_environment():
    # Keep Slurm/device visibility and ordinary runtime configuration. No key
    # values are inspected, logged, persisted, or forwarded into the container.
    allowed = {"PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH"}
    return {
        key: value
        for key, value in os.environ.items()
        if key in allowed or key.startswith(("SLURM_", "ROCR_VISIBLE_", "HIP_VISIBLE_", "CUDA_VISIBLE_"))
    }


def _record_stage(run_dir, record):
    with (run_dir / "stages.jsonl").open("a") as output:
        output.write(json.dumps(record) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _run(stage, argv, *, run_dir, env):
    print(f"[{stage}] {shlex.join(map(str, argv))}", flush=True)
    started = time.monotonic()
    record = {
        "stage": stage,
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "argv": list(map(str, argv)),
    }
    # Persist before launch: a killed or hung process still leaves its active stage.
    _record_stage(run_dir, {**record, "event": "started"})
    result = None
    try:
        with (run_dir / f"{stage}.log").open("w") as log:
            result = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"Stage {stage} failed ({result.returncode}); see {stage}.log")
        if stage == "lm-head-98k" and "OOM" in (run_dir / f"{stage}.log").read_text():
            raise RuntimeError("The benchmark caught an out-of-memory error; qualification failed")
    except BaseException as error:
        _record_stage(
            run_dir,
            {
                **record,
                "event": "failed",
                "elapsed_seconds": time.monotonic() - started,
                "returncode": result.returncode if result is not None else None,
                "error_type": type(error).__name__,
            },
        )
        raise
    _record_stage(
        run_dir,
        {
            **record,
            "event": "succeeded",
            "elapsed_seconds": time.monotonic() - started,
            "returncode": result.returncode,
        },
    )


def run_in_allocation(*, config_path, work_dir, run_dir, expected_commit=None):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU preparation must run through Slurm")
    qualification, training = load_configs(config_path)
    if os.environ.get("SLURM_JOB_PARTITION") != qualification["allocation"]["partition"]:
        raise RuntimeError("This job requires the configured MI350 partition")
    if not run_dir.is_relative_to(work_dir):
        raise ValueError("run-dir must be within work-dir so the container can write it")
    work_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "manifest.json").exists():
        raise RuntimeError("Use a fresh run directory; existing evidence will not be overwritten")
    if shutil.disk_usage(work_dir).free < qualification["container"]["minimum_free_bytes"]:
        raise RuntimeError("Less than 120 GiB filesystem space is free; also check the lab quota")
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if not expected_commit or commit != expected_commit:
        raise RuntimeError("Checkout changed since batch-file review; render and review a new batch file")
    if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Use a clean, committed checkout")
    manifest = {
        "started_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "job_id": os.environ["SLURM_JOB_ID"],
        "commit": commit,
        "qualification": qualification,
        "training_proposal": training,
        "qualification_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "scope": qualification.get("scope"),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    env = _public_environment()
    env["APPTAINER_CACHEDIR"] = str(work_dir / "cache" / "apptainer")
    env["APPTAINER_TMPDIR"] = str(work_dir / "tmp")
    for key in ("APPTAINER_CACHEDIR", "APPTAINER_TMPDIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    digest = qualification["container"]["uri"].split("sha256:")[1]
    image_dir = work_dir / "containers"
    image_dir.mkdir(exist_ok=True)
    sif = image_dir / f"{digest}.sif"
    sandbox = image_dir / f"{digest}.sandbox"
    if not sif.exists():
        temporary = sif.with_suffix(".partial.sif")
        _run(
            "pull",
            ["apptainer", "pull", "--force", str(temporary), qualification["container"]["uri"]],
            run_dir=run_dir,
            env=env,
        )
        temporary.rename(sif)
    # Sandbox extraction makes image files owned by the ordinary cluster user.
    # In particular, /root/Megatron-LM is then readable without fakeroot/subuids.
    if not (sandbox / ".miles-ready").exists():
        if sandbox.exists():
            raise RuntimeError("Incomplete sandbox exists; inspect it and choose a fresh work directory")
        _run("sandbox", ["apptainer", "build", "--sandbox", str(sandbox), str(sif)], run_dir=run_dir, env=env)
        (sandbox / ".miles-ready").touch()
    _run_gpu_checks(
        sandbox=sandbox,
        work_dir=work_dir,
        run_dir=run_dir,
        config_path=config_path,
        qualification=qualification,
        training=training,
        env=env,
    )
    (run_dir / "completed.json").write_text(
        json.dumps(
            {
                "completed_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "full_training_qualified": False,
            }
        )
        + "\n"
    )


def _run_gpu_checks(*, sandbox, work_dir, run_dir, config_path, qualification, training, env):
    runtime_dir = None
    if qualification.get("mode") == "rl_smoke":
        runtime_dir = run_dir / "ray"
        runtime_dir.mkdir(exist_ok=True)
    prefix = container_prefix(
        image=sandbox,
        repo=ROOT,
        work_dir=work_dir,
        isolate_processes=qualification.get("mode") == "rl_smoke",
        runtime_dir=runtime_dir,
        cache_namespace=qualification["allocation"]["partition"],
    )
    if qualification.get("mode") == "compatibility":
        for phase in ("torch-only", "after-engine-import"):
            _run(
                phase,
                prefix
                + [
                    "python3",
                    "-m",
                    "experiments.tcs_async_rl.compatibility_probe",
                    "--phase",
                    phase,
                    "--run-dir",
                    str(run_dir),
                ],
                run_dir=run_dir,
                env=env,
            )
        return
    if qualification.get("mode") == "group_inference":
        _run_group_check(
            prefix=prefix, work_dir=work_dir, run_dir=run_dir, config_path=config_path, training=training, env=env
        )
        return
    if qualification.get("mode") == "rl_smoke":
        _run("training-help", prefix + ["python3", "train_async.py", "--help"], run_dir=run_dir, env=env)
        _run(
            "rl-smoke",
            prefix
            + [
                "python3",
                "-m",
                "experiments.tcs_async_rl.rl_smoke",
                "--config",
                str(config_path),
                "--work-dir",
                str(work_dir),
                "--run-dir",
                str(run_dir),
            ],
            run_dir=run_dir,
            env=env,
        )
        return
    module = ["python3", "-m", "experiments.tcs_async_rl.gpu_qualification"]
    _run(
        "imports",
        prefix + module + ["imports", "--run-dir", str(run_dir), "--config", str(config_path)],
        run_dir=run_dir,
        env=env,
    )
    gpus = qualification["allocation"]["gpus"]
    _run(
        "collective",
        prefix
        + [
            "torchrun",
            "--standalone",
            f"--nproc-per-node={gpus}",
            "-m",
            "experiments.tcs_async_rl.gpu_qualification",
            "collective",
        ],
        run_dir=run_dir,
        env=env,
    )
    _run("training-help", prefix + ["python3", "train_async.py", "--help"], run_dir=run_dir, env=env)
    model = training["model"]
    model_dir = work_dir / "models" / f"Qwen3.5-4B-{model['revision']}"
    _run(
        "model",
        prefix + ["hf", "download", model["repo_id"], "--revision", model["revision"], "--local-dir", str(model_dir)],
        run_dir=run_dir,
        env=env,
    )
    _run(
        "lm-head-correctness",
        prefix + ["python3", "-m", "pytest", "-q", "tests/fast-gpu/test_fused_lm_head.py"],
        run_dir=run_dir,
        env=env,
    )
    geometry = model_inputs(training, base_dir=config_path.parent)
    actor = training["miles_args"]
    _run(
        "lm-head-98k",
        prefix
        + [
            "python3",
            "tests/manual/bench_fused_lm_head.py",
            "--rows",
            str(actor["seq_length"]),
            "--hidden",
            str(geometry["hidden_size"]),
            "--vocab-shard",
            str(geometry["vocab_size"] // actor["tensor_model_parallel_size"]),
            "--tiles",
            str(actor["log_probs_chunk_size"]),
            "--skip-torch",
        ],
        run_dir=run_dir,
        env=env,
    )
    _run(
        "inference",
        prefix
        + module
        + ["inference", "--model-dir", str(model_dir), "--config", str(config_path), "--run-dir", str(run_dir)],
        run_dir=run_dir,
        env=env,
    )


def _run_group_check(*, prefix, work_dir, run_dir, config_path, training, env):
    _run(
        "engine-import",
        prefix
        + [
            "python3",
            "-c",
            "from sglang.srt.entrypoints.engine import Engine; from aiter.jit.core import get_user_jit_dir; import torch; x=torch.ones((16,16), device='cuda', dtype=torch.bfloat16); y=x@x; torch.cuda.synchronize(); assert bool((y==16).all()); print('AITER_JIT_DIR=' + get_user_jit_dir()); print('Engine import and BF16 matmul OK')",
        ],
        run_dir=run_dir,
        env=env,
    )
    model = training["model"]
    model_dir = work_dir / "models" / f"Qwen3.5-4B-{model['revision']}"
    _run(
        "model",
        prefix + ["hf", "download", model["repo_id"], "--revision", model["revision"], "--local-dir", str(model_dir)],
        run_dir=run_dir,
        env=env,
    )
    _run(
        "group-inference",
        prefix
        + [
            "python3",
            "-m",
            "experiments.tcs_async_rl.group_inference",
            "--model-dir",
            str(model_dir),
            "--config",
            str(config_path),
            "--run-dir",
            str(run_dir),
        ],
        run_dir=run_dir,
        env=env,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-in-allocation", action="store_true")
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    paths = {
        "config_path": args.config.resolve(),
        "work_dir": args.work_dir.resolve(),
        "run_dir": args.run_dir.resolve(),
    }
    if args.run_in_allocation:
        run_in_allocation(**paths, expected_commit=args.expected_commit)
    else:
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        print(render_job(repo=ROOT, expected_commit=commit, **paths), end="")


if __name__ == "__main__":
    main()
