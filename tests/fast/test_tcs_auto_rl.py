import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experiments.tcs_async_rl.auto_rl import qualify_medium, run
from experiments.tcs_async_rl.group_inference import sampling_parameters
from experiments.tcs_async_rl.rl_submission import submit_once
from experiments.tcs_async_rl.watch_rl import consumed_rollouts, training_evidence

BASE = Path(__file__).resolve().parents[2] / "experiments/tcs_async_rl"


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        config = json.loads((BASE / "rl_smoke_v7_medium.json").read_text())
        training = json.loads((BASE / config["training_config"]).read_text())
        for key in ("training_config", "container_config"):
            config[key] = str(BASE / config[key])
        config["input_config"] = "input-config.json"
        self.config = self.root / "config.json"
        self.write("config.json", config)
        self.write("input.json", {"test_fixture": "synthetic, no private data"})
        self.write(
            "input-config.json",
            {
                "training_config": config["training_config"],
                "input_sha256": hashlib.sha256((self.root / "input.json").read_bytes()).hexdigest(),
            },
        )
        self.write(
            "watcher-complete.json",
            {
                "scheduler": {"state": "COMPLETED", "grading_returncode": 0},
                "hf_results": "https://example.invalid/private-archive",
            },
        )
        scores = {
            "control-reference": 7,
            "control-empty": 0,
            "control-irrelevant": 0,
            **{f"sample-{i:02d}": (5 if i % 2 else 0) for i in range(32)},
        }
        self.grades = {
            "generation_sampling": sampling_parameters(training),
            "scored_responses": 32,
            "incomplete_or_empty_final_answers": 0,
            "scores_by_case": scores,
            "model": training["judge"]["primary_model"],
            "reasoning_effort": training["judge"]["reasoning_effort"],
            "grades": [
                {"identity": name, "swapped": order, "candidate": {"score": score}, "reference": {"score": 7}}
                for name, score in scores.items()
                for order in (False, True)
            ],
        }
        self.write("group-grades.json", self.grades)
        rows = [{"sample_index": i, "final_answer": "synthetic answer"} for i in range(32)]
        (self.root / "responses.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
        self.write(
            "sampling-support-control.json",
            {
                "output_ids": [1] * 16,
                "meta_info": {
                    "output_token_sampling_mask": [[1]] * 16,
                    "output_token_sampling_logprobs": [[0.0]] * 16,
                },
            },
        )
        self.write("resolved-inference-mode.json", {"speculative_algorithm": None})

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def test_complete_varied_group_can_qualify_without_any_submission(self):
        result = qualify_medium(self.root, self.config)
        self.assertEqual(result["input_sha256"], hashlib.sha256((self.root / "input.json").read_bytes()).hexdigest())

    def test_missing_answers_uniform_scores_sampling_and_bad_reference_block(self):
        original = copy.deepcopy(self.grades)
        for mutate in (
            lambda g: g.update(incomplete_or_empty_final_answers=1),
            lambda g: g.update(scored_responses=31),
            lambda g: g["generation_sampling"].update(top_p=1.0),
            lambda g: g["scores_by_case"].update({f"sample-{i:02d}": 0 for i in range(32)}),
            lambda g: g["grades"][0]["reference"].update(score=5),
            lambda g: g["grades"].pop(),
        ):
            candidate = copy.deepcopy(original)
            mutate(candidate)
            self.write("group-grades.json", candidate)
            with self.assertRaises(ValueError):
                qualify_medium(self.root, self.config)

    def test_mismatched_input_or_response_indices_block(self):
        self.write("input.json", {"another": "question"})
        with self.assertRaisesRegex(ValueError, "hash"):
            qualify_medium(self.root, self.config)

    def test_failed_gate_never_starts_service_or_submits_gpu_job(self):
        cli = SimpleNamespace(
            config=self.config,
            expected_commit="frozen",
            diagnostic_dir=self.root,
            group_job_id="123",
            remote_diagnostic_dir="/remote",
            state_repo=self.root,
            run_dir=self.root,
        )
        with patch("experiments.tcs_async_rl.auto_rl.subprocess.check_output", return_value="frozen"), patch(
            "experiments.tcs_async_rl.auto_rl.wait_for_group"
        ), patch("experiments.tcs_async_rl.auto_rl.qualify_medium", side_effect=ValueError("gate failed")), patch(
            "experiments.tcs_async_rl.auto_rl.start_service"
        ) as service, patch(
            "experiments.tcs_async_rl.auto_rl.stage_and_submit"
        ) as submit:
            with self.assertRaises(ValueError):
                run(cli)
            service.assert_not_called()
            submit.assert_not_called()


class SubmissionTests(unittest.TestCase):
    def test_restart_reuses_receipt_and_never_submits_twice(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            invoke = Mock(return_value=subprocess.CompletedProcess([], 0, "12345\n", ""))
            first = submit_once(root, root / "job.py", invoke=invoke)
            second = submit_once(root, root / "job.py", invoke=invoke)
            self.assertEqual(first, second)
            invoke.assert_called_once()

    def test_lost_submission_response_requires_reconciliation(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            invoke = Mock(side_effect=subprocess.TimeoutExpired("sbatch", 60))
            with self.assertRaises(subprocess.TimeoutExpired):
                submit_once(root, root / "job.py", invoke=invoke)
            with self.assertRaisesRegex(RuntimeError, "Reconcile"):
                submit_once(root, root / "job.py", invoke=invoke)
            invoke.assert_called_once()


class EvidenceTests(unittest.TestCase):
    def test_duplicate_rank_logs_are_not_extra_updates_and_zero_loss_is_valid(self):
        lines = []
        for i in range(2):
            lines.extend([f"train op=train_step rollout={i} step=0 attempt=0 outcome=NORMAL valid_step=true"] * 2)
            lines.append(f"step {i}: {{'train/pg_loss': 0.0, 'train/grad_norm': 1.5}}")
        evidence = training_evidence("\n".join(lines))
        self.assertEqual(len(evidence["valid_updates"]), 2)
        self.assertFalse(evidence["skipped_or_discarded_updates"])
        self.assertTrue(all(x["finite"] for x in evidence["metrics"].values()))
        bad = training_evidence(
            "train op=train_step rollout=0 step=0 attempt=0 outcome=NORMAL valid_step=false\n"
            "step 0: {'train/pg_loss': nan, 'train/grad_norm': 0.0}"
        )
        self.assertEqual(len(bad["skipped_or_discarded_updates"]), 1)
        self.assertFalse(bad["metrics"][0]["finite"])

    def test_nested_weight_spans_are_counted_and_export_hash_is_checked(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "rollout_data").mkdir()
            path = root / "rollout_data/rollout-0.jsonl"
            row = {
                "reward": {"reward": 5},
                "optimizer_rollout_id": 0,
                "response_length": 10,
                "weight_versions": [{"spans": [{"version": "1", "abs_start": 10, "abs_end": 20}]}],
            }
            path.write_text(json.dumps(row) + "\n")
            (root / "rollout-export.json").write_text(
                json.dumps(
                    [
                        {
                            "path": "/remote/rollout_data/rollout-0.jsonl",
                            "samples": 1,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                    ]
                )
            )
            group = consumed_rollouts(root)[0]
            self.assertEqual(group["decode_tokens_by_weight_version"], {"1": 10})
            self.assertFalse(group["varied_rewards"])
            path.write_text(json.dumps(row) + " \n")
            with self.assertRaisesRegex(ValueError, "differs"):
                consumed_rollouts(root)


if __name__ == "__main__":
    unittest.main()
