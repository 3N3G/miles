import asyncio
import copy
import gzip
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experiments.tcs_async_rl.grade_group import Budget, grade_one
from experiments.tcs_async_rl.reward_hook import question_key
from experiments.tcs_async_rl.reward_service import MultiScorer
from experiments.tcs_async_rl.rl_archive import archive_chunks, copy_trace_batch, read_index
from experiments.tcs_async_rl.rl_artifacts import file_sha256, retain_latest_checkpoint, save_generated_sample
from experiments.tcs_async_rl.rl_inputs import question_set_key, read_bundle
from experiments.tcs_async_rl.rl_smoke import native_argv
from experiments.tcs_async_rl.watch_rl import acceptance, consumed_rollouts, run_training_evidence, watch

BASE = Path(__file__).resolve().parents[2] / "experiments/tcs_async_rl"


def question(index):
    return {
        "problem": f"question {index}",
        "answer": "reference",
        "rubric": "Total: 7 points\n1. [7 pt] correct proof",
        "difficulty": "easy",
        "cohort": "STOC_2023",
        "paper_id": f"paper{index}",
        "node_id": "node_0",
    }


class EasyDataTests(unittest.TestCase):
    def test_budget_timeout_does_not_mean_nonfinite_training(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "rl-smoke.log").write_text(
                "train op=train_step rollout=0 step=0 attempt=0 outcome=NORMAL valid_step=true\n"
                "step 0: {'train/pg_loss': 0.1, 'train/grad_norm': 0.18}\n"
            )
            (root / "checkpoint-inventory.json").write_text(
                json.dumps({"tracker": "0", "iterations": [{"files": 4, "bytes": 100}]})
            )
            with patch(
                "experiments.tcs_async_rl.watch_rl.consumed_rollouts",
                return_value=[{"samples": 32, "varied_rewards": True}],
            ):
                result = acceptance(
                    root, {"state": "TIMEOUT", "exit_code": "0:0"}, 40, minimum_varied_groups=1, retained_checkpoints=1
                )
            self.assertEqual(result["stop_reason"], "walltime_limit")
            self.assertEqual(result["completed_updates"], 1)
            self.assertTrue(result["checks"]["finite_step_metrics"])
            self.assertFalse(result["checks"]["optimizer_updates_recorded"])
            self.assertFalse(result["runtime_acceptance_passed"])

    def test_monitor_retries_terminal_archive_backpressure(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            config = {"overrides": {"rm_url": "http://127.0.0.1:18495/score"}}
            with patch("experiments.tcs_async_rl.watch_rl.load_configs", return_value=(config, {})), patch(
                "experiments.tcs_async_rl.watch_rl.remote_state", return_value={"state": "TIMEOUT"}
            ), patch("experiments.tcs_async_rl.watch_rl.record"), patch(
                "experiments.tcs_async_rl.watch_rl.sync_outputs", return_value=0
            ), patch(
                "experiments.tcs_async_rl.watch_rl.finish", side_effect=[RuntimeError("disk pressure"), None]
            ) as finish, patch(
                "experiments.tcs_async_rl.watch_rl.time.sleep"
            ):
                watch(
                    job_id="123",
                    config_path=root / "config.json",
                    directory=root,
                    remote_directory=Path("/remote"),
                    checkout=Path("/checkout"),
                    state_repo=root,
                    children=[Mock(), Mock()],
                )
            self.assertEqual(finish.call_count, 2)
            retry = json.loads((root / "archive-retry.json").read_text())
            self.assertEqual(retry["job_id"], "123")
            self.assertEqual(retry["scheduler"]["state"], "TIMEOUT")
            self.assertEqual(retry["phase"], "finalization_pending")

    def test_finalizer_recovers_after_monitor_error_without_finished_marker(self):
        from experiments.tcs_async_rl.finalize_rl_logs import finalize

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "submission.json").write_text('{"job_id":"123"}')
            (root / "approved-config.json").write_text("{}")
            original = '{"phase":"stopped","error_type":"RuntimeError"}\n'
            (root / "handoff-state.json").write_text(original)
            cli = SimpleNamespace(
                run_dir=root,
                job_id="123",
                config=root / "config.json",
                remote_run_dir=Path("/remote"),
                remote_checkout=Path("/checkout"),
                state_repo=root,
            )
            with patch("experiments.tcs_async_rl.finalize_rl_logs.load_configs", return_value=({}, {})), patch(
                "experiments.tcs_async_rl.finalize_rl_logs.controller_alive", return_value=False
            ), patch(
                "experiments.tcs_async_rl.finalize_rl_logs.remote_state", return_value={"state": "TIMEOUT"}
            ), patch(
                "experiments.tcs_async_rl.finalize_rl_logs.finish"
            ) as finish, patch(
                "experiments.tcs_async_rl.finalize_rl_logs.upload_run",
                return_value="https://huggingface.co/datasets/owner/data/commit/" + "a" * 40,
            ), patch(
                "experiments.tcs_async_rl.finalize_rl_logs.time.sleep", side_effect=AssertionError("Unexpected wait")
            ):
                finalize(cli)
            self.assertEqual(finish.call_args.kwargs["scheduler"]["state"], "TIMEOUT")
            self.assertEqual((root / "analysis-before-log-reader-repair/handoff-state.json").read_text(), original)
            self.assertTrue((root / "native-log-finalizer-complete.json").exists())

    def test_optimizer_metrics_are_read_from_the_native_subprocess_log(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "slurm-123.log").write_text("[rl-smoke] apptainer exec ...\n")
            (root / "rl-smoke.log").write_text(
                "train op=train_step rollout=0 step=0 attempt=0 outcome=NORMAL valid_step=true\n"
                "step 0: {'train/pg_loss': 0.1, 'train/grad_norm': 0.18}\n"
            )
            evidence = run_training_evidence(root)
            self.assertEqual(evidence["valid_updates"], [[0, 0, 0]])
            self.assertTrue(evidence["metrics"][0]["finite"])

    def test_weight_check_excludes_only_frozen_vision_tensors(self):
        config = json.loads((BASE / "rl_easy_overfit_v9.json").read_text())
        values = config["overrides"]
        self.assertTrue(values["check_weight_update_equal"])
        self.assertEqual(values["check_weight_update_skip_list"], ["visual."])
        self.assertEqual(
            native_argv({"check_weight_update_skip_list": values["check_weight_update_skip_list"]}),
            ["--check-weight-update-skip-list", "visual."],
        )

    def test_only_the_pinned_twenty_easy_eligible_questions_are_accepted(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "bundle.json"
            rows = [question(i) for i in range(20)]
            training = json.loads((BASE / "config.sampling_repair_v2.json").read_text())
            bundle = {
                "questions": rows,
                "dataset_revision": training["data"]["revision"],
                "question_set_key": question_set_key(rows),
            }

            def check(value):
                path.write_text(json.dumps(value))
                return read_bundle(path, expected_sha256=file_sha256(path), training=training)

            self.assertEqual(len(check(bundle)[0]), 20)
            for field, value in [("cohort", "FOCS_2026"), ("difficulty", "hard")]:
                bad = copy.deepcopy(bundle)
                bad["questions"][0][field] = value
                with self.assertRaises(ValueError):
                    check(bad)
            bad = copy.deepcopy(bundle)
            bad["questions"][1] = bad["questions"][0]
            with self.assertRaises(ValueError):
                check(bad)


class MultiQuestionTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_unfinished_judge_verdicts_get_bounded_token_retries(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            config = json.loads((BASE / "judge_group_v2_inference_repair.json").read_text())
            judge = {"primary_model": "dummy", "reasoning_effort": "medium", "max_completion_tokens": 8192}
            verdict = {key: [{"item": 1, "satisfied": True, "reason": "proved"}] for key in ("A", "B")}

            def receipt(payload, **kwargs):
                capped = payload["max_completion_tokens"] < 32768
                return {
                    "http_status": 200,
                    "body": {
                        "usage": {"prompt_tokens": 50, "completion_tokens": 10},
                        "choices": [
                            {
                                "finish_reason": "length" if capped else "stop",
                                "message": {"content": "" if capped else json.dumps(verdict)},
                            }
                        ],
                    },
                }

            kwargs = dict(
                identity="test",
                candidate="answer",
                row=question(0),
                swapped=False,
                judge=judge,
                config=config,
                prompt="judge",
                key="dummy",
                budget=Budget(root, config),
                semaphore=asyncio.Semaphore(1),
                directory=root,
            )
            with patch("experiments.tcs_async_rl.grade_group.request_chat", side_effect=receipt) as client:
                result = await grade_one(**kwargs)
                self.assertEqual(result["candidate"]["score"], 7)
                self.assertEqual(
                    [c.args[0]["max_completion_tokens"] for c in client.call_args_list], [8192, 16384, 32768]
                )
                await grade_one(**kwargs)
                self.assertEqual(client.call_count, 3)
            self.assertEqual(len(list(root.glob("*-attempt0.json"))), 3)
            self.assertEqual(len(list(root.glob("length-retries.jsonl"))), 1)

    async def test_question_routing_isolated_cache_and_shared_spending_cap(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            rows = [question(0), question(1)]
            path = root / "bundle.json"
            path.write_text(json.dumps({"questions": rows, "question_set_key": question_set_key(rows)}))
            with patch(
                "experiments.tcs_async_rl.reward_service.load_credentials", return_value={"OPENAI_API_KEY": "dummy"}
            ):
                service = MultiScorer(
                    input_file=path,
                    directory=root / "judging",
                    config_path=BASE / "judge_group_v1.json",
                    key_file=root / "unused",
                )
            first, second = service.scorers.values()
            self.assertIs(first.budget, second.budget)
            self.assertIs(first.semaphore, second.semaphore)

            async def grade(**kwargs):
                value = 0 if kwargs["row"]["problem"] == "question 0" else 7
                return {"candidate": {"score": value}, "reference": {"score": 7}}

            with patch("experiments.tcs_async_rl.reward_service.grade_one", side_effect=grade):
                results = await asyncio.gather(
                    *(
                        service.score({"question_key": question_key(row), "response": "thinking</think>same answer"})
                        for row in rows
                    )
                )
            self.assertEqual([r["reward"] for r in results], [0, 7])
            with self.assertRaises(ValueError):
                await service.score({"question_key": "foreign", "response": "answer"})


class ArchiveTests(unittest.TestCase):
    def test_low_vm_space_blocks_submission_before_any_cluster_action(self):
        from experiments.tcs_async_rl.auto_rl import stage_and_submit

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with patch(
                "experiments.tcs_async_rl.rl_archive.shutil.disk_usage", return_value=SimpleNamespace(free=4 * 2**30)
            ), patch("experiments.tcs_async_rl.auto_rl.remote") as remote:
                with self.assertRaisesRegex(RuntimeError, "VM monitoring needs"):
                    stage_and_submit(
                        config_path=root / "missing.json",
                        source_commit="unused",
                        grades={},
                        directory=root,
                        work_dir=Path("/remote"),
                        checkout=Path("/checkout"),
                        remote_run=Path("/run"),
                    )
            remote.assert_not_called()

    def test_trace_transfer_stays_under_the_vm_batch_budget(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "rollout_data").mkdir()
            files = [
                ("rollout_data/rollout-0.jsonl.gz", 50 * 2**20),
                ("rollout_data/completed_samples/a.generated.json.gz", 50 * 2**20),
            ]
            answers = [
                subprocess.CompletedProcess([], 0, json.dumps(files), ""),
                subprocess.CompletedProcess([], 0, "", ""),
            ]
            with patch("experiments.tcs_async_rl.rl_archive.subprocess.run", side_effect=answers), patch(
                "experiments.tcs_async_rl.rl_archive.shutil.disk_usage", return_value=SimpleNamespace(free=2 * 2**30)
            ):
                remaining = copy_trace_batch(Path("/remote"), root)
            self.assertEqual(remaining, 1)
            self.assertEqual((root / "trace-transfer.list").read_text().strip(), "rollout_data/rollout-0.jsonl.gz")

    def test_low_space_defers_transfer_without_deleting_or_copying(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            result = subprocess.CompletedProcess(
                [], 0, json.dumps([("rollout_data/rollout-0.jsonl.gz", 20 * 2**20)]), ""
            )
            with patch("experiments.tcs_async_rl.rl_archive.subprocess.run", return_value=result) as invoke, patch(
                "experiments.tcs_async_rl.rl_archive.shutil.disk_usage", return_value=SimpleNamespace(free=512 * 2**20)
            ):
                with self.assertRaisesRegex(RuntimeError, "Trace transfer deferred"):
                    copy_trace_batch(Path("/remote"), root)
            self.assertEqual(invoke.call_count, 1)

    def test_checkpoint_retention_never_leaves_this_runs_directory(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            save = root / "run/checkpoints"
            save.mkdir(parents=True)
            (save.parent / "submission.json").write_text('{"job_id":"123"}')
            old = save / "iter_0000004"
            current = save / "iter_0000009"
            old.mkdir()
            current.mkdir()
            (current / "state").write_text("keep")
            foreign = root / "foreign"
            foreign.mkdir()
            (foreign / "state").write_text("keep")
            (save / "iter_0000000").symlink_to(foreign, target_is_directory=True)
            retain_latest_checkpoint(SimpleNamespace(save=str(save)), 9, str(current), None)
            self.assertFalse(old.exists())
            self.assertTrue((current / "state").exists())
            self.assertTrue((foreign / "state").exists())
            with self.assertRaises(ValueError):
                retain_latest_checkpoint(SimpleNamespace(save=str(save)), 10, str(foreign), None)

    def make_trace(self, root):
        folder = root / "rollout_data"
        folder.mkdir()
        path = folder / "rollout-0.jsonl.gz"
        row = {
            "reward": {"reward": 5},
            "optimizer_rollout_id": 0,
            "weight_versions": [],
            "response_length": 2,
            "metadata": {"question_key": "q"},
            "status": "completed",
        }
        with gzip.open(path, "wt") as stream:
            stream.write(json.dumps(row) + "\n")
        (root / "rollout-export.json").write_text(
            json.dumps([{"path": str(path), "samples": 1, "sha256": file_sha256(path)}])
        )
        return path

    def test_verified_upload_can_evict_only_immutable_local_copies_and_reuse_summary(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = self.make_trace(root)
            before = consumed_rollouts(root)
            invoke = Mock(
                return_value=subprocess.CompletedProcess(
                    [], 0, "url=https://huggingface.co/datasets/owner/data/commit/" + "a" * 40, " "
                )
            )
            archive_chunks(root, invoke=invoke)
            self.assertFalse(path.exists())
            self.assertTrue(read_index(root))
            self.assertEqual(before, consumed_rollouts(root))
            index = read_index(root)
            index["rollout_data/rollout-0.jsonl.gz"]["sha256"] = "wrong"
            (root / "archive-index.json").write_text(json.dumps(index))
            with self.assertRaises(ValueError):
                consumed_rollouts(root)

    def test_failed_or_ambiguous_upload_keeps_every_local_trace(self):
        for code, stdout in [(1, ""), (0, "no immutable receipt")]:
            with tempfile.TemporaryDirectory() as name:
                root = Path(name)
                path = self.make_trace(root)
                invoke = Mock(return_value=subprocess.CompletedProcess([], code, stdout, ""))
                with self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
                    archive_chunks(root, invoke=invoke)
                self.assertTrue(path.exists())
                self.assertFalse(read_index(root))

    def test_compressed_generated_output_roundtrips_exact_policy_replay_fields(self):
        with tempfile.TemporaryDirectory() as name:
            sample = {
                "response": "think</think>answer",
                "tokens": [1, 2, 3],
                "rollout_log_probs": [-0.1, -0.2],
                "status": "completed",
                "metadata": {"question_key": "q"},
                "response_length": 2,
            }
            path = save_generated_sample(str(Path(name) / "rollout-0.pt"), sample, compress=True)
            with gzip.open(path, "rt") as stream:
                row = json.load(stream)
            self.assertEqual(row["tokens"], sample["tokens"])
            self.assertEqual(row["rollout_log_probs"], sample["rollout_log_probs"])


if __name__ == "__main__":
    unittest.main()
