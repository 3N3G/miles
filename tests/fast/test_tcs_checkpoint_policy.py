import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.tcs_async_rl.amd_preflight import load_configs
from experiments.tcs_async_rl.checkpoint_policy import (
    FINAL_CHECKPOINT_HOOK,
    request_final_checkpoint,
    single_checkpoint,
    storage_required,
)
from experiments.tcs_async_rl.rl_smoke import native_argv, resolved_args


class CheckpointPolicyTests(unittest.TestCase):
    def config(self, run):
        config, training = load_configs(Path("experiments/tcs_async_rl/rl_easy_overfit_v15.json"))
        config["overrides"].update(
            custom_megatron_before_train_step_hook_path=FINAL_CHECKPOINT_HOOK,
            save_interval=None,
            save_trigger_sentinel=str(run / "checkpoint-request"),
        )
        return config, training

    def test_final_request_preserves_raw_outputs_and_never_requests_second_checkpoint(self):
        with tempfile.TemporaryDirectory() as name:
            run = Path(name)
            root = run / "checkpoints"
            root.mkdir()
            args = SimpleNamespace(
                save=str(root),
                save_interval=None,
                save_trigger_sentinel=str(run / "checkpoint-request"),
                num_rollout=105,
            )
            request_final_checkpoint(args, 103, 0, None, None, None)
            self.assertFalse(Path(args.save_trigger_sentinel).exists())
            request_final_checkpoint(args, 104, 0, None, None, None)
            self.assertTrue(Path(args.save_trigger_sentinel).exists())
            Path(args.save_trigger_sentinel).unlink()
            (root / "latest_checkpointed_iteration.txt").write_text("102")
            request_final_checkpoint(args, 104, 0, None, None, None)
            self.assertFalse(Path(args.save_trigger_sentinel).exists())
            self.assertEqual((root / "latest_checkpointed_iteration.txt").read_text(), "102")

    def test_checkpoint_peak_follows_verified_save_policy(self):
        run = Path("/work/runs/continuation")
        config, training = self.config(run)
        self.assertTrue(single_checkpoint(config, run))
        self.assertEqual(storage_required(config, run, 1000), 16000 + 20 * 2**30)
        args, _ = resolved_args(config, training, work_dir=Path("/work"), run_dir=run, run_id="test")
        self.assertNotIn("--save-interval", native_argv(args))
        self.assertEqual(args["save_trigger_sentinel"], str(run / "checkpoint-request"))
        self.assertEqual(args["num_rollout"], 105)
        old = copy.deepcopy(config)
        del old["overrides"]["custom_megatron_before_train_step_hook_path"]
        self.assertEqual(storage_required(old, run, 1000), 30000 + 20 * 2**30)

    def test_reject_periodic_saves_and_foreign_sentinels_before_lowering_space_guard(self):
        run = Path("/work/runs/continuation")
        config, _ = self.config(run)
        for key, value in (("save_interval", 5), ("save_trigger_sentinel", "/work/runs/source/checkpoint-request")):
            broken = copy.deepcopy(config)
            broken["overrides"][key] = value
            with self.assertRaises(ValueError):
                storage_required(broken, run, 1000)


if __name__ == "__main__":
    unittest.main()
