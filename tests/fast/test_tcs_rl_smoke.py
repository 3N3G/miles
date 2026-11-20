import copy
import json
import unittest
from pathlib import Path

from experiments.tcs_async_rl.amd_preflight import container_prefix, load_configs
from experiments.tcs_async_rl.rl_smoke import native_argv, resolved_args, validate_group_gate


class RLSmokeTests(unittest.TestCase):
    def gate(self):
        return {
            "scores_by_case": {
                "control-reference": 7,
                "control-empty": 0,
                "control-irrelevant": 0,
                "sample-00": 0,
                "sample-01": 1,
            },
            "scored_responses": 2,
            "incomplete_or_empty_final_answers": 0,
        }

    def test_gate_rejects_uniform_incomplete_and_failed_controls(self):
        validate_group_gate(self.gate(), 2)
        for mutate in (
            lambda g: g["scores_by_case"].update({"sample-01": 0}),
            lambda g: g.update(incomplete_or_empty_final_answers=1),
            lambda g: g["scores_by_case"].update({"control-reference": 6}),
            lambda g: g.update(scored_responses=1),
        ):
            grades = self.gate()
            mutate(grades)
            with self.assertRaises(ValueError):
                validate_group_gate(grades, 2)

    def test_native_args_are_fully_resolved_and_keep_thinking_markers(self):
        config, training = load_configs(Path("experiments/tcs_async_rl/rl_smoke_v1.json"))
        original = copy.deepcopy(training)
        args, name = resolved_args(
            config, training, work_dir=Path("/work"), run_dir=Path("/work/run"), run_id="test-id"
        )
        argv = native_argv(args)
        self.assertNotIn("--rollout-skip-special-tokens", argv)
        self.assertEqual(json.loads(argv[argv.index("--apply-chat-template-kwargs") + 1]), {"enable_thinking": True})
        self.assertEqual(args["rollout_max_response_len"], 90000)
        self.assertEqual(args["n_samples_per_prompt"], 32)
        self.assertEqual(args["rollout_batch_size"] * args["n_samples_per_prompt"], args["global_batch_size"])
        self.assertEqual(args["num_rollout"], 2)
        self.assertEqual(args["wandb_mode"], "offline")
        self.assertNotIn("--wandb-key", argv)
        self.assertIn(name, args["ref_load"])
        self.assertEqual(training, original)

    def test_private_pid_namespace_is_mandatory_for_rl_container(self):
        common = dict(image=Path("/work/image"), repo=Path("/repo"), work_dir=Path("/work"))
        ordinary = container_prefix(**common)
        isolated = container_prefix(**common, isolate_processes=True)
        self.assertNotIn("--pid", ordinary)
        self.assertIn("--pid", isolated)
        self.assertTrue(any(value.startswith("TCS_HOST_PID_NAMESPACE=pid:[") for value in isolated))


if __name__ == "__main__":
    unittest.main()
