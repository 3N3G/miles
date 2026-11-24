import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experiments.tcs_async_rl.amd_preflight import load_configs
from experiments.tcs_async_rl.resource_guard import (
    duration_seconds,
    resource_limits,
    stop_reason,
    validated_stop,
    watch,
)
from experiments.tcs_async_rl.watch_rl import acceptance


class ResourceGuardTests(unittest.TestCase):
    def test_guard_cannot_touch_an_allocation_with_a_different_owner(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "judging").mkdir()
            (root / "judging/budget.jsonl").write_text(json.dumps({"usd_delta": 22}) + "\n")
            (root / "submission.json").write_text(json.dumps({"job_id": "123", "session_uuid": "session"}))
            (root / "approved-config.json").write_text("{}")
            cli = SimpleNamespace(
                config=root / "config.json",
                directory=root,
                remote_directory=Path("/work/run"),
                job_id="123",
                session_uuid="session",
            )
            limits = {"elapsed_seconds": 41400, "spent_reserved_usd": 22, "free_bytes": 100}
            transport = Mock(return_value=SimpleNamespace(stdout="Avail\n1000\n"))
            with patch(
                "experiments.tcs_async_rl.resource_guard.load_configs", return_value=({"job_name": "owned"}, {})
            ), patch("experiments.tcs_async_rl.resource_guard.resource_limits", return_value=limits), patch(
                "experiments.tcs_async_rl.resource_guard.subprocess.check_output", return_value="user owner\n"
            ), patch(
                "experiments.tcs_async_rl.resource_guard.remote_state",
                return_value={"state": "RUNNING", "elapsed": "00:00:01"},
            ), patch(
                "experiments.tcs_async_rl.resource_guard.remote", transport
            ), patch(
                "experiments.tcs_async_rl.resource_guard.allocation", side_effect=ValueError("wrong owner")
            ):
                with self.assertRaisesRegex(ValueError, "wrong owner"):
                    watch(cli)
            self.assertEqual([call.args[0][0] for call in transport.call_args_list], ["df"])
            self.assertFalse((root / "resource-stop-request.json").exists())

    def test_limits_use_the_approved_walltime_and_preserve_save_reserves(self):
        base = Path("experiments/tcs_async_rl")
        config, training = load_configs(base / "rl_easy_overfit_v17.json")
        config = copy.deepcopy(config)
        config.update(
            judge_config="judge_group_reference_high_20261124.json",
            resource_stop={"walltime_margin_seconds": 1800, "judge_stop_usd": 22},
        )
        config["allocation"]["walltime"] = "12:00:00"
        config["overrides"].update(
            stop_after_save_trigger=True, single_checkpoint=False, save_trigger_sentinel="/work/run/checkpoint-request"
        )
        limits = resource_limits(config, training, base, Path("/work/run"))
        self.assertEqual(limits["elapsed_seconds"], 41400)
        self.assertEqual(limits["spent_reserved_usd"], 22)
        self.assertGreater(limits["free_bytes"], 120_000_000_000)
        self.assertEqual(duration_seconds("1-01:02:03"), 90123)
        for values, reason in (
            ([41400, 1, 10**12], "walltime_save_margin"),
            ([1, 22, 10**12], "judge_budget"),
            ([1, 1, 0], "checkpoint_storage"),
            ([1, 1, 10**12], None),
        ):
            self.assertEqual(
                stop_reason(
                    limits, dict(zip(("elapsed_seconds", "spent_reserved_usd", "free_bytes"), values, strict=True))
                ),
                reason,
            )

    def test_stop_receipt_cannot_forge_ownership_or_a_crossed_limit(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "approved-config.json").write_text("{}")
            (root / "submission.json").write_text(json.dumps({"job_id": "123", "session_uuid": "session"}))
            limits = {"elapsed_seconds": 41400, "spent_reserved_usd": 22, "free_bytes": 100}
            marker = {
                "job_id": "123",
                "session_uuid": "session",
                "approved_config_sha256": hashlib.sha256(b"{}").hexdigest(),
                "limits": limits,
                "observed": {"elapsed_seconds": 10, "spent_reserved_usd": 22, "free_bytes": 1000},
                "reason": "judge_budget",
            }
            with patch("experiments.tcs_async_rl.resource_guard.load_configs", return_value=({}, {})), patch(
                "experiments.tcs_async_rl.resource_guard.resource_limits", return_value=limits
            ):
                for change in (
                    {},
                    {"job_id": "other"},
                    {"session_uuid": "other"},
                    {"observed": {"elapsed_seconds": 10, "spent_reserved_usd": 1, "free_bytes": 1000}},
                ):
                    (root / "resource-stop-request.json").write_text(json.dumps({**marker, **change}))
                    if change:
                        with self.assertRaises(ValueError):
                            validated_stop(root, root / "config.json", Path("/remote"))
                    else:
                        self.assertEqual(
                            validated_stop(root, root / "config.json", Path("/remote"))["reason"], "judge_budget"
                        )

    def test_a_planned_stop_requires_all_completed_steps_and_their_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "training-finished.json").write_text("{}")
            (root / "checkpoint-inventory.json").write_text(
                json.dumps({"tracker": "110", "iterations": [{"files": 4, "bytes": 100}]})
            )
            evidence = {
                "valid_updates": [[109, 0, 0], [110, 0, 0]],
                "skipped_or_discarded_updates": [],
                "metrics": {109: {"finite": True}, 110: {"finite": True}},
            }
            groups = [{"samples": 128, "varied_rewards": True}] * 2
            with patch("experiments.tcs_async_rl.watch_rl.run_training_evidence", return_value=evidence), patch(
                "experiments.tcs_async_rl.watch_rl.consumed_rollouts", return_value=groups
            ):
                kwargs = dict(
                    minimum_varied_groups=1,
                    retained_checkpoints=1,
                    initial_updates=109,
                    expected_samples_per_update=128,
                )
                self.assertFalse(
                    acceptance(root, {"state": "COMPLETED", "exit_code": "0:0"}, 10000, **kwargs)[
                        "runtime_acceptance_passed"
                    ]
                )
                good = acceptance(
                    root,
                    {"state": "COMPLETED", "exit_code": "0:0"},
                    10000,
                    approved_stop={"reason": "judge_budget"},
                    **kwargs,
                )
                self.assertTrue(good["runtime_acceptance_passed"])
                self.assertEqual(good["stop_reason"], "judge_budget")
                self.assertEqual(good["target_updates"], 10000)
                (root / "checkpoint-inventory.json").write_text(
                    json.dumps({"tracker": "109", "iterations": [{"files": 4, "bytes": 100}]})
                )
                self.assertFalse(
                    acceptance(
                        root,
                        {"state": "COMPLETED", "exit_code": "0:0"},
                        10000,
                        approved_stop={"reason": "judge_budget"},
                        **kwargs,
                    )["runtime_acceptance_passed"]
                )


if __name__ == "__main__":
    unittest.main()
