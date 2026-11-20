"""CPU checks for allocation safety, secret exclusion, and context accounting."""

import ast
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.tcs_async_rl.amd_preflight import (
    CONFIG,
    ROOT,
    _public_environment,
    _run,
    container_prefix,
    load_configs,
    render_job,
    run_in_allocation,
)
from experiments.tcs_async_rl.gpu_qualification import engine_arguments, validate_lengths


class TestAMDPreflight(unittest.TestCase):
    def test_render_has_real_limits_and_no_submission(self):
        rendered = render_job(
            repo=ROOT, work_dir=Path("/work/tcs"), run_dir=Path("/work/tcs/run"), expected_commit="abc123"
        )
        ast.parse(rendered)
        self.assertIn("#SBATCH --nodes=1", rendered)
        self.assertIn("#SBATCH --time=02:00:00", rendered)
        self.assertIn("#SBATCH --partition=mi3508x", rendered)
        self.assertNotIn("#SBATCH --mem", rendered)
        self.assertNotIn("sbatch", rendered)

    def test_rejects_login_node_execution_before_creating_files(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Slurm"):
                run_in_allocation(config_path=CONFIG, work_dir=Path("/nonexistent"), run_dir=Path("/nonexistent"))

    def test_rejects_other_partition(self):
        with patch.dict(os.environ, {"SLURM_JOB_ID": "1", "SLURM_JOB_PARTITION": "mi3501x"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "partition"):
                run_in_allocation(config_path=CONFIG, work_dir=Path("/nonexistent"), run_dir=Path("/nonexistent"))

    def test_does_not_forward_credentials_or_injected_commands(self):
        fake = {
            "PATH": "/usr/bin",
            "SLURM_JOB_ID": "123",
            "OPENAI_API_KEY": "dummy",
            "GEMINI_API_KEY": "dummy",
            "APPTAINERENV_LD_PRELOAD": "dummy",
        }
        with patch.dict(os.environ, fake, clear=True):
            self.assertEqual(_public_environment(), {"PATH": "/usr/bin", "SLURM_JOB_ID": "123"})

    def test_container_uses_checked_out_source_and_work_disk(self):
        argv = container_prefix(image=Path("/work/sandbox"), repo=ROOT, work_dir=Path("/work/tcs"))
        self.assertIn("--cleanenv", argv)
        self.assertIn("--no-home", argv)
        self.assertIn(f"PYTHONPATH={ROOT}:/root/Megatron-LM:/sgl-workspace/sglang/python", argv)
        self.assertIn("HF_HOME=/work/tcs/cache/huggingface", argv)
        self.assertIn("AITER_JIT_DIR=/work/tcs/cache/aiter/sandbox", argv)

    def test_clean_container_receives_only_explicit_job_and_device_metadata(self):
        fake = {
            "SLURM_JOB_ID": "123",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "OPENAI_API_KEY": "dummy",
            "WANDB_API_KEY": "dummy",
        }
        with patch.dict(os.environ, fake, clear=True):
            argv = container_prefix(image=Path("/work/sandbox"), repo=ROOT, work_dir=Path("/work/tcs"))
        self.assertIn("SLURM_JOB_ID=123", argv)
        self.assertIn("CUDA_VISIBLE_DEVICES=0,1", argv)
        self.assertFalse(any("API_KEY" in value for value in argv))

    def test_gpu_compilation_caches_are_separate_across_partitions(self):
        common = dict(image=Path("/work/sandbox"), repo=ROOT, work_dir=Path("/work/tcs"))
        older = container_prefix(**common, cache_namespace="mi2101x")
        newer = container_prefix(**common, cache_namespace="mi3508x")
        for name in ("AITER_JIT_DIR", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR"):
            first = next(value for value in older if value.startswith(name + "="))
            second = next(value for value in newer if value.startswith(name + "="))
            self.assertNotEqual(first, second)

    def test_long_case_reaches_exact_context_boundary(self):
        qualification, training = load_configs()
        self.assertLess(
            qualification["inference"]["short_prompt_tokens"] + qualification["inference"]["generated_tokens"],
            training["miles_args"]["sglang_context_length"],
        )
        arguments = engine_arguments(Path("/work/model"), qualification, training)
        self.assertEqual(arguments["speculative_algorithm"], "EAGLE")
        self.assertEqual(arguments["tp_size"], 2)
        self.assertFalse(arguments["allow_auto_truncate"])

    def test_detects_silent_truncation(self):
        with self.assertRaisesRegex(RuntimeError, "input length"):
            validate_lengths(
                {"meta_info": {"prompt_tokens": 4096, "completion_tokens": 32}},
                prompt_tokens=98272,
                generated_tokens=32,
            )

    def test_detects_short_decode(self):
        with self.assertRaisesRegex(RuntimeError, "decode length"):
            validate_lengths(
                {"meta_info": {"prompt_tokens": 98272, "completion_tokens": 31}},
                prompt_tokens=98272,
                generated_tokens=32,
            )

    def test_caught_benchmark_oom_still_fails_qualification(self):
        def caught_oom(argv, **kwargs):
            kwargs["stdout"].write("fused_lm_head: OOM\n")
            return subprocess.CompletedProcess(argv, 0)

        with tempfile.TemporaryDirectory() as temporary:
            with patch("subprocess.run", side_effect=caught_oom):
                with self.assertRaisesRegex(RuntimeError, "out-of-memory"):
                    _run("lm-head-98k", ["benchmark"], run_dir=Path(temporary), env={})

    def test_subprocess_failure_stops_qualification(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("subprocess.run", return_value=subprocess.CompletedProcess([], 2)):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    _run("imports", ["check"], run_dir=Path(temporary), env={})

    def test_stage_start_is_durable_before_subprocess_starts(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary) / "stages.jsonl"

            def inspect_started(argv, **kwargs):
                records = [json.loads(line) for line in journal.read_text().splitlines()]
                self.assertEqual([record["event"] for record in records], ["started"])
                return subprocess.CompletedProcess(argv, 0)

            with patch("subprocess.run", side_effect=inspect_started):
                _run("imports", ["check"], run_dir=Path(temporary), env={})
            records = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual([record["event"] for record in records], ["started", "succeeded"])

    def test_launch_exception_is_recorded_as_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("subprocess.run", side_effect=FileNotFoundError):
                with self.assertRaises(FileNotFoundError):
                    _run("imports", ["missing"], run_dir=Path(temporary), env={})
            records = [json.loads(line) for line in (Path(temporary) / "stages.jsonl").read_text().splitlines()]
            self.assertEqual([record["event"] for record in records], ["started", "failed"])
            self.assertEqual(records[-1]["error_type"], "FileNotFoundError")


if __name__ == "__main__":
    unittest.main()
