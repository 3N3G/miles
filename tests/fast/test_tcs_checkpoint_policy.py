import ast
import asyncio
import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.tcs_async_rl.amd_preflight import load_configs
from experiments.tcs_async_rl.checkpoint_policy import single_checkpoint, storage_required
from experiments.tcs_async_rl.rl_smoke import native_argv, resolved_args
from miles.utils.checkpoint_policy import should_save_checkpoint


class CheckpointPolicyTests(unittest.TestCase):
    def test_actual_async_checkpoint_block_keeps_one_save_and_preserves_default_cadence(self):
        # Execute the driver's actual checkpoint block without its optional GPU
        # imports. This checks the caller, epoch predicate and save ordering.
        source = ast.parse(Path("train_async.py").read_text())
        train = next(node for node in source.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "train")
        loop = next(
            node
            for node in train.body
            if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "rollout_id"
        )
        start = next(
            i
            for i, node in enumerate(loop.body)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "external_save" for target in node.targets)
        )
        function = ast.parse("async def checkpoint_step(rollout_id): pass").body[0]
        function.body = loop.body[start : start + 2]
        misc = ast.parse(Path("miles/utils/misc.py").read_text())
        predicate = next(
            node
            for node in misc.body
            if isinstance(node, ast.FunctionDef) and node.name == "should_run_periodic_action"
        )

        async def simulate(single, deadline):
            with tempfile.TemporaryDirectory() as name:
                root = Path(name) / "checkpoints"
                root.mkdir()
                args = SimpleNamespace(
                    single_checkpoint=single,
                    save=str(root),
                    save_interval=105 if single else 5,
                    save_trigger_sentinel=str(root.parent / "request"),
                    num_rollout=105,
                    use_critic=False,
                )
                events = []

                async def save_data(step):
                    events.append(("data", step))

                async def save_model(model, step, force_sync):
                    events.append(("model", step, force_sync))
                    (root / "latest_checkpointed_iteration.txt").write_text(str(step))

                namespace = dict(
                    args=args,
                    os=os,
                    num_rollout_per_epoch=20,
                    rollout_executor=SimpleNamespace(save=save_data),
                    actor_model=object(),
                    save_training_model=save_model,
                    should_save_checkpoint=should_save_checkpoint,
                )
                module = ast.Module(body=[predicate, function], type_ignores=[])
                exec(compile(ast.fix_missing_locations(module), "actual-checkpoint-block", "exec"), namespace)
                for step in range(85, 105):
                    if step == deadline:
                        Path(args.save_trigger_sentinel).touch()
                    await namespace["checkpoint_step"](step)
                return events

        normal = asyncio.run(simulate(False, None))
        self.assertEqual([event[1] for event in normal if event[0] == "model"], [89, 94, 99, 104])
        self.assertEqual(asyncio.run(simulate(True, None)), [("data", 104), ("model", 104, True)])
        self.assertEqual(asyncio.run(simulate(True, 99)), [("data", 99), ("model", 99, True)])

    def config(self, run):
        config, training = load_configs(Path("experiments/tcs_async_rl/rl_easy_overfit_v15.json"))
        config["overrides"].update(
            single_checkpoint=True, save_interval=105, save_trigger_sentinel=str(run / "checkpoint-request")
        )
        return config, training

    def test_one_save_overrides_epoch_boundary_and_preserves_completed_deadline_checkpoint(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / "checkpoints"
            root.mkdir()
            args = SimpleNamespace(single_checkpoint=True, save=str(root), num_rollout=105)
            self.assertFalse(should_save_checkpoint(args, 99, external_save=False, periodic_save=True))
            self.assertTrue(should_save_checkpoint(args, 104, external_save=False, periodic_save=True))
            self.assertTrue(should_save_checkpoint(args, 102, external_save=True, periodic_save=False))
            (root / "latest_checkpointed_iteration.txt").write_text("102")
            self.assertFalse(should_save_checkpoint(args, 104, external_save=False, periodic_save=True))
            with self.assertRaisesRegex(RuntimeError, "already has"):
                should_save_checkpoint(args, 104, external_save=True, periodic_save=True)
            self.assertEqual((root / "latest_checkpointed_iteration.txt").read_text(), "102")

    def test_normal_checkpoint_cadence_is_unchanged(self):
        args = SimpleNamespace(single_checkpoint=False)
        for external in (False, True):
            for periodic in (False, True):
                self.assertEqual(
                    should_save_checkpoint(args, 99, external_save=external, periodic_save=periodic),
                    external or periodic,
                )

    def test_checkpoint_peak_follows_verified_save_policy_and_positive_native_interval(self):
        run = Path("/work/runs/continuation")
        config, training = self.config(run)
        self.assertTrue(single_checkpoint(config, run))
        self.assertEqual(storage_required(config, run, 1000), 16000 + 20 * 2**30)
        args, _ = resolved_args(config, training, work_dir=Path("/work"), run_dir=run, run_id="test")
        argv = native_argv(args)
        self.assertIn("--single-checkpoint", argv)
        self.assertEqual(argv[argv.index("--save-interval") + 1], "105")
        self.assertEqual(args["save_trigger_sentinel"], str(run / "checkpoint-request"))
        self.assertEqual(args["num_rollout"], 105)
        old = copy.deepcopy(config)
        del old["overrides"]["single_checkpoint"]
        self.assertEqual(storage_required(old, run, 1000), 30000 + 20 * 2**30)

    def test_reject_invalid_intervals_and_foreign_sentinels_before_lowering_space_guard(self):
        run = Path("/work/runs/continuation")
        config, _ = self.config(run)
        for key, value in (
            ("save_interval", 0),
            ("save_interval", None),
            ("save_trigger_sentinel", "/work/runs/source/request"),
        ):
            broken = copy.deepcopy(config)
            broken["overrides"][key] = value
            with self.assertRaises(ValueError):
                storage_required(broken, run, 1000)


if __name__ == "__main__":
    unittest.main()
