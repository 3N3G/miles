"""Startup transfer checks must accept trained checkpoints and reject corruption."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from miles.backends.training_utils.weight_update.initial_check import check_initial_weight_update


class Engine:
    def __init__(self):
        self.weights = {"language": 1.0, "head": 2.0, "visual.": 3.0}
        self.snapshot = dict(self.weights)
        self.actions = []

    async def check_weights(self, *, action, allow_quant_error, selector, skip_list):
        self.actions.append(action)
        if action == "snapshot":
            self.snapshot = dict(self.weights)
        elif action == "reset_tensors":
            self.weights = {k: v if k in skip_list else -99.0 for k, v in self.weights.items()}
        elif action == "compare":
            if any(v != self.snapshot[k] for k, v in self.weights.items() if k not in skip_list):
                raise RuntimeError("Weight comparison failed")


def args(*, start=25, enabled=True):
    return SimpleNamespace(
        check_weight_update_equal=enabled,
        start_rollout_id=start,
        check_weight_update_allow_quant_error=False,
        check_weight_update_selector="all",
        check_weight_update_skip_list=["visual."],
    )


class InitialWeightCheckTests(unittest.IsolatedAsyncioTestCase):
    async def test_fresh_initialization_still_compares_to_original_hf_weights(self):
        engine, publish = Engine(), AsyncMock()
        await check_initial_weight_update(args(start=0), inference_controller=engine, republish_weights=publish)
        self.assertEqual(engine.actions, ["compare"])
        publish.assert_not_awaited()

    async def test_restored_checkpoint_uses_snapshot_reset_and_rebroadcast(self):
        engine = Engine()
        restored = {"language": 1.125, "head": 2.25, "visual.": 3.0}
        engine.weights.update(restored)
        with self.assertRaises(RuntimeError):
            await engine.check_weights(action="compare", allow_quant_error=False, selector="all", skip_list=[])
        engine.actions.clear()

        async def publish():
            self.assertEqual(engine.weights, {"language": -99.0, "head": -99.0, "visual.": 3.0})
            engine.weights.update(restored)

        await check_initial_weight_update(args(), inference_controller=engine, republish_weights=publish)
        self.assertEqual(engine.actions, ["snapshot", "reset_tensors", "compare"])
        self.assertEqual(engine.weights, restored)

    async def test_missing_tensor_in_second_transfer_still_fails(self):
        engine = Engine()
        engine.weights.update(language=1.125, head=2.25)

        async def incomplete_publish():
            engine.weights["language"] = 1.125

        with self.assertRaises(RuntimeError):
            await check_initial_weight_update(
                args(), inference_controller=engine, republish_weights=incomplete_publish
            )

    async def test_unrequested_checks_do_not_reset_or_republish(self):
        engine, publish = Engine(), AsyncMock()
        await check_initial_weight_update(args(enabled=False), inference_controller=engine, republish_weights=publish)
        self.assertEqual(engine.actions, [])
        publish.assert_not_awaited()
