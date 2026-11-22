import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from experiments.tcs_async_rl.diagnose_rl import question_comparisons, trace_summary
from experiments.tcs_async_rl.live_review import build_review
from experiments.tcs_async_rl.native_metrics import (
    compare_wandb,
    publish_audit,
    read_native_metrics,
    sync_live_audit,
    sync_terminal_audit,
)
from experiments.tcs_async_rl.sampled_entropy import summarize as summarize_sampled_entropy


class DiagnosticTests(unittest.TestCase):
    def test_detailed_review_flags_wrong_update_direction_and_compares_only_repeated_questions(self):
        native = [
            {"optimizer_step": 25, "train/lr-pg_0": 1e-6, "train/update_direction/surrogate_gain": -0.1},
            {"optimizer_step": 26, "train/lr-pg_0": 1e-6},
            {"optimizer_step": 27, "train/lr-pg_0": 1e-6},
        ]
        groups = [
            {
                "rollout_ids": [step],
                "by_question": {key: {"scores": [reward], "lengths": [100], "capped": 0}},
                "example_file": "raw.json",
                "decode_tokens_by_weight_version": {},
            }
            for step, key, reward in [(25, "a", 1), (26, "b", 6), (27, "a", 2)]
        ]
        result = build_review(native, groups, [])
        self.assertEqual(result["completed_updates"], 3)
        self.assertEqual(result["flags_to_inspect"]["negative_observed_surrogate_gain"], [25])
        self.assertEqual(result["within_continuation_matched_visits"]["mean_change"], 1)
        self.assertEqual(result["within_continuation_matched_visits"]["questions"], 1)

    def test_resuming_filters_metric_history_restored_from_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "events").mkdir()
            (root / "approved-config.json").write_text(json.dumps({"resume": {"iteration": 24}}))
            for source, metric in [
                ("actor_cell00000_rank00000.jsonl", "train/pg_loss"),
                ("rollout_executor.jsonl", "rollout/episode_raw_reward"),
            ]:
                events = [{"type": "metric", "rollout_id": step, "metrics": {metric: 1.0}} for step in [0, 24, 25]]
                (root / "events" / source).write_text("\n".join(json.dumps(e) for e in events))
            self.assertEqual([r["optimizer_step"] for r in read_native_metrics(root)], [25])

    def test_sampled_surprisal_keeps_answer_and_token_weighting_separate_and_matches_questions(self):
        rows = [
            dict(
                optimizer_step=step,
                question_key=question,
                paper_id=question,
                mean_sampled_log_probability=-surprisal,
                response_tokens=tokens,
                tokens_with_support_size_one=0,
                reward=1,
            )
            for step, question, surprisal, tokens in [(0, "a", 1, 1), (0, "a", 3, 3), (1, "b", 8, 1), (2, "a", 2.5, 1)]
        ]
        result = summarize_sampled_entropy(rows)
        self.assertEqual(result["groups"][0]["answer_mean_sampled_surprisal_nats"], 2)
        self.assertEqual(result["groups"][0]["token_weighted_sampled_surprisal_nats"], 2.5)
        self.assertEqual(len(result["repeated_questions"]), 1)
        self.assertEqual(result["repeated_questions"][0]["change_nats"], 0.5)

    def test_live_reader_waits_for_weight_publication_and_handles_a_nonzero_resume_step(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "events").mkdir()
            actor = root / "events/actor_cell00000_rank00000.jsonl"
            rollout = root / "events/rollout_executor.jsonl"
            actor.write_text(
                json.dumps({"type": "metric", "rollout_id": 25, "metrics": {"train/step": 25, "train/pg_loss": 0.1}})
                + "\n"
            )
            rollout.write_text(
                json.dumps(
                    {
                        "type": "metric",
                        "rollout_id": 25,
                        "metrics": {"rollout/step": 25, "rollout/episode_raw_reward": 3},
                    }
                )
                + "\n"
            )
            self.assertEqual(read_native_metrics(root, allow_pending=True, require_publication=True), [])
            with actor.open("a") as stream:
                stream.write(
                    json.dumps({"type": "metric", "rollout_id": 25, "metrics": {"perf/update_weights_time": 0.1}})
                    + "\n"
                )
                stream.write('{"incomplete":')
            rows = read_native_metrics(root, allow_pending=True, require_publication=True)
            self.assertEqual([row["optimizer_step"] for row in rows], [25])

    def test_live_publisher_keeps_running_then_closes_failed_and_does_not_resend_verified_steps(self):
        native = [{"optimizer_step": 25, "rollout/episode_raw_reward": 3.0, "train/pg_loss": 0.1}]
        remote = Mock(state="running", url="https://wandb.example/run")
        remote.scan_history.return_value = [{"audit/optimizer_step": 25, "audit/rollout/episode_raw_reward": 3.0}]
        sdk = Mock()
        sdk.Api.return_value.run.return_value = remote
        sdk.init.return_value.resumed = True
        sdk.init.return_value.summary = {}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / "run"
            root.mkdir()
            (root / "resolved-config.json").write_text(
                json.dumps({"args": {"wandb_team": "team", "wandb_project": "project"}, "run_id": "id"})
            )
            with patch.dict("sys.modules", {"wandb": sdk}), patch(
                "experiments.tcs_async_rl.native_metrics.read_native_metrics", return_value=native
            ):
                receipt = sync_live_audit(root, root, "RUNNING")
                self.assertEqual(receipt["verified_points"], 1)
                self.assertFalse(sdk.Settings.call_args.kwargs["x_update_finish_state"])
                self.assertEqual(sdk.init.call_args.kwargs["id"], "id-native")
                remote.scan_history.assert_called_with(
                    keys=["audit/optimizer_step", "audit/rollout/episode_raw_reward"], use_cache=False
                )
                sync_live_audit(root, root, "RUNNING")
                self.assertEqual(sdk.init.call_count, 1)
                remote.state = "failed"
                sync_live_audit(root, root, "TIMEOUT")
                self.assertTrue(sdk.Settings.call_args.kwargs["x_update_finish_state"])
                sdk.init.return_value.finish.assert_called_with(exit_code=1)

    def test_live_publisher_does_not_resend_an_upload_when_history_indexing_exceeds_its_readback_window(self):
        native = [{"optimizer_step": 25, "rollout/episode_raw_reward": 3.0, "train/pg_loss": 0.1}]
        remote = Mock(state="running", url="https://wandb.example/run")
        sdk = Mock()
        sdk.Api.return_value.run.return_value = remote
        sdk.init.return_value.summary = {}
        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / "run"
            root.mkdir()
            (root / "resolved-config.json").write_text(
                json.dumps({"args": {"wandb_team": "team", "wandb_project": "project"}, "run_id": "id"})
            )
            with patch.dict("sys.modules", {"wandb": sdk}), patch(
                "experiments.tcs_async_rl.native_metrics.read_native_metrics", return_value=native
            ), patch("experiments.tcs_async_rl.native_metrics.time.sleep"):
                remote.scan_history.return_value = []
                with self.assertRaisesRegex(RuntimeError, "0/1 steps"):
                    sync_live_audit(root, root, "RUNNING")
                remote.scan_history.return_value = [
                    {"audit/optimizer_step": 25, "audit/rollout/episode_raw_reward": 3.0}
                ]
                receipt = sync_live_audit(root, root, "RUNNING")
                self.assertEqual(receipt["verified_points"], 1)
                self.assertEqual(sdk.init.call_count, 1)
                sdk.init.return_value.log.assert_called_once()

    def test_native_events_recover_reward_axis_after_offline_writer_collision(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "events").mkdir()
            actor, rollout = [], []
            for step, reward in enumerate([1.0, 6.0]):
                actor.extend(
                    [
                        {"type": "metric", "rollout_id": step, "metrics": {"train/step": step, "train/pg_loss": 0.1}},
                        {
                            "type": "metric",
                            "rollout_id": step,
                            "metrics": {"rollout/step": step, "rollout/raw_reward": reward},
                        },
                    ]
                )
                rollout.append(
                    {
                        "type": "metric",
                        "rollout_id": step,
                        "metrics": {"rollout/step": step, "rollout/episode_raw_reward": reward},
                    }
                )
            (root / "events/actor_cell00000_rank00000.jsonl").write_text("\n".join(map(json.dumps, actor)))
            path = root / "events/rollout_executor.jsonl"
            path.write_text("\n".join(map(json.dumps, rollout)))
            native = read_native_metrics(root)
            # A colliding W&B writer supplied the second reward with step zero.
            history = [
                {"rollout/step": 0, "rollout/episode_raw_reward": 1.0},
                {"rollout/step": 0, "rollout/episode_raw_reward": 6.0},
            ]
            audit = compare_wandb(native, history, metric="rollout/episode_raw_reward", step_key="rollout/step")
            self.assertEqual(audit["missing_or_misassigned_native_steps"], [1])
            self.assertEqual(native[1]["rollout/episode_raw_reward"], 6.0)
            rollout[1]["metrics"]["rollout/step"] = 0
            path.write_text("\n".join(map(json.dumps, rollout)))
            with self.assertRaisesRegex(ValueError, "axis disagrees"):
                read_native_metrics(root)

    def test_question_matching_does_not_confuse_a_harder_second_pass_with_regression(self):
        def group(step, key, reward):
            return {
                "optimizer_step": step,
                "question_key": key,
                "paper_id": key,
                "mean_reward": reward,
                "cap_rate": 0,
                "mean_tokens": 100,
            }

        groups = [group(0, "easy", 7), group(1, "hard", 1), group(2, "hard", 2)]
        pairs = question_comparisons(groups)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["difference"], 1)
        self.assertEqual(pairs[0]["first_step"], 1)

    def test_trace_validation_rejects_selected_tokens_outside_the_replayed_support(self):
        sample = {
            "response_length": 2,
            "tokens": [99, 3, 4],
            "rollout_log_probs": [0.0, -0.5],
            "loss_mask": None,
            "rollout_sampling_mask": {"encoding": "csr", "ids": [3, 4, 5], "offsets": [0, 1, 3]},
            "weight_versions": [{"spans": [{"version": "0", "abs_start": 1, "abs_end": 3}]}],
            "status": "completed",
            "response": "thinking</think>answer",
            "final_answer": "answer",
        }
        result = trace_summary(sample)
        self.assertEqual(result["active_loss_tokens"], 2)
        self.assertEqual(result["tokens_with_support_size_one"], 1)
        sample["rollout_sampling_mask"]["ids"][1] = 8
        with self.assertRaises(AssertionError):
            trace_summary(sample)

    def test_corrected_metrics_use_one_writer_and_a_separate_explicit_axis(self):
        run = Mock()
        publish_audit([{"optimizer_step": 3, "rollout/episode_raw_reward": 2.0}], run)
        run.define_metric.assert_any_call("audit/*", step_metric="audit/optimizer_step")
        run.log.assert_called_once_with({"audit/optimizer_step": 3, "audit/rollout/episode_raw_reward": 2.0})

    def test_wandb_readback_retries_indexing_without_republishing_or_changing_terminal_state(self):
        native = [{"optimizer_step": 0, "rollout/episode_raw_reward": 2.0}]
        digest = hashlib.sha256(json.dumps(native, sort_keys=True).encode()).hexdigest()
        before = Mock(state="failed", summary={"audit/native_metrics_sha256": digest})
        delayed = Mock(state="failed")
        delayed.scan_history.return_value = [{"audit/optimizer_step": None}]
        indexed = Mock(state="failed", url="https://wandb.example/run")
        indexed.scan_history.return_value = [{"audit/optimizer_step": 0, "audit/rollout/episode_raw_reward": 2.0}]
        sdk = Mock()
        sdk.Api.return_value.run.side_effect = [before, delayed, indexed]
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "resolved-config.json").write_text(
                json.dumps({"args": {"wandb_team": "team", "wandb_project": "project"}, "run_id": "id"})
            )
            with patch.dict("sys.modules", {"wandb": sdk}), patch(
                "experiments.tcs_async_rl.native_metrics.read_native_metrics", return_value=native
            ), patch("experiments.tcs_async_rl.native_metrics.time.sleep") as sleep:
                receipt = sync_terminal_audit(root, root)
            sdk.init.assert_not_called()
            sleep.assert_called_once_with(2)
            self.assertEqual(receipt["verified_points"], 1)
            self.assertEqual(receipt["state_after"], "failed")


if __name__ == "__main__":
    unittest.main()
