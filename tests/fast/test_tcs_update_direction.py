import ast
import datetime
import json
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from experiments.tcs_async_rl.gsm8k_artifacts import write_json
from experiments.tcs_async_rl.gsm8k_control import CONFIG, resolved_args
from miles.backends.training_utils.metrics.update_direction import aggregate_update_direction, compute_update_direction


def _distributed_direction(rank, rendezvous, directory, empty_second_shard):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=15),
    )
    try:
        changes = [0.1, -0.1] if rank == 0 else [-0.5]
        advantages = [1.0, -1.0] if rank == 0 else [1.0]
        masks = [torch.tensor([0.0 if rank == 1 and empty_second_shard else 1.0]) for _ in changes]
        local = compute_update_direction(
            [torch.zeros(1) for _ in changes],
            [torch.tensor([x]) for x in changes],
            advantages=[torch.tensor([x]) for x in advantages],
            behavior=[torch.zeros(1) for _ in changes],
            loss_masks=masks,
            eps_clip=0.2,
            eps_clip_high=0.28,
            allow_empty=True,
        )
        result = aggregate_update_direction(local, dp_group=dist.group.WORLD, dp_size=2)
        (Path(directory) / f"rank-{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


class UpdateDirectionRepairTests(unittest.TestCase):
    def test_both_approved_arms_pass_real_direction_validation_without_changing_denominators(self):
        # Exercise the production pure validator without importing Megatron's
        # GPU runtime and plugin tree on this CPU host.
        source = Path(__file__).resolve().parents[2] / "miles/utils/arguments.py"
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate_update_direction_args")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        validate = namespace[function.name]
        config = json.loads(CONFIG.read_text())
        for arm in ("reference", "cispo"):
            args = resolved_args(config, arm=arm, work_dir=Path("/w"), run_dir=Path("/w/run"), run_id=arm)
            parsed = types.SimpleNamespace(
                **{
                    "train_backend": "megatron",
                    "loss_type": "policy_loss",
                    "indep_dp": False,
                    "use_rollout_logprobs": False,
                    "skip_actor_forward_only": False,
                    **args,
                }
            )
            validate(parsed)
            self.assertEqual(parsed.use_rollout_logprobs, arm == "cispo")
            self.assertEqual(parsed.actor_num_gpus_per_node, 2)
            for setting, value in (("indep_dp", True), ("context_parallel_size", 2), ("pipeline_model_parallel_size", 2)):
                bad = types.SimpleNamespace(**{**vars(parsed), setting: value})
                with self.assertRaises(ValueError):
                    validate(bad)

    def test_two_data_shards_match_the_full_batch_and_allow_a_fully_masked_shard(self):
        for empty in (False, True):
            with self.subTest(empty_second_shard=empty), tempfile.TemporaryDirectory() as directory:
                rendezvous = str(Path(directory) / "rendezvous")
                mp.spawn(_distributed_direction, args=(rendezvous, directory, empty), nprocs=2, join=True)
                results = [json.loads((Path(directory) / f"rank-{rank}.json").read_text()) for rank in (0, 1)]
                self.assertEqual(results[0], results[1])
                result = results[0]
                self.assertEqual(result["samples"], 2 if empty else 3)
                self.assertEqual(result["negative_advantage_samples"], 1)
                self.assertAlmostEqual(result["surrogate_gain"], 0.1 if empty else -0.1, places=6)
                self.assertAlmostEqual(result["positive_advantage_mean_logprob_change"], 0.1 if empty else -0.2, places=6)

    def test_observation_detaches_inputs_and_detects_good_bad_and_zero_changes(self):
        for change in (0.1, -0.1, 0.0):
            before = torch.zeros(2, requires_grad=True)
            after = torch.tensor([change, -change], requires_grad=True)
            advantages = torch.tensor([1.0, -1.0], requires_grad=True)
            behavior = torch.zeros(2, requires_grad=True)
            result = compute_update_direction(
                [before],
                [after],
                advantages=[advantages],
                behavior=[behavior],
                loss_masks=[torch.ones(2)],
                eps_clip=0.2,
                eps_clip_high=0.28,
            )
            self.assertAlmostEqual(result["surrogate_gain"], change, places=6)
            for tensor in (before, after, advantages, behavior):
                self.assertIsNone(tensor.grad)
        empty = compute_update_direction([], [], advantages=[], behavior=[], loss_masks=[], eps_clip=0.2, eps_clip_high=0.28, allow_empty=True)
        with self.assertRaises(ValueError):
            aggregate_update_direction(empty, dp_group=None, dp_size=1)

    def test_native_argument_metadata_can_serialize_dtype_but_metric_json_stays_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arguments.json"
            write_json(path, {"params_dtype": torch.bfloat16}, default=str)
            self.assertEqual(json.loads(path.read_text())["params_dtype"], "torch.bfloat16")
            with self.assertRaises(TypeError):
                write_json(path, {"params_dtype": torch.bfloat16})
            with self.assertRaises(ValueError):
                write_json(path, {"metric": float("nan")}, default=str)


if __name__ == "__main__":
    unittest.main()
