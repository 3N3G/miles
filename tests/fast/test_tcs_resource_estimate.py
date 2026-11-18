import json
import unittest
from pathlib import Path

from experiments.tcs_async_rl.resource_estimate import (
    compute_scenarios,
    kv_bytes_per_token,
    memory_estimates,
    model_inputs,
    model_state_bytes,
)


class ResourceEstimateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2] / "experiments/tcs_async_rl/config.proposed.json"
        cls.config = json.loads(path.read_text())
        cls.inputs = model_inputs(cls.config, base_dir=path.parent)

    def test_model_geometry_has_three_gib_kv_without_mtp(self):
        inputs = self.inputs
        self.assertEqual(kv_bytes_per_token(inputs, mtp=False), 32768)
        self.assertEqual(kv_bytes_per_token(inputs, mtp=False) * 98304, 3 * 2**30)
        self.assertEqual(kv_bytes_per_token(inputs, mtp=True) * 98304, 3.375 * 2**30)

    def test_data_parallelism_shards_optimizer_not_model_weights(self):
        # A two-way shard retains the 6-byte weight/gradient portion, and halves 12-byte Adam/master state.
        self.assertEqual(model_state_bytes(100, tp=1, dp_cp=1), 1800)
        self.assertEqual(model_state_bytes(100, tp=1, dp_cp=2), 1200)
        self.assertEqual(model_state_bytes(100, tp=2, dp_cp=1), 900)
        with self.assertRaises(ValueError):
            model_state_bytes(100, tp=0, dp_cp=1)

    def test_batch_and_live_engine_capacity_are_separate(self):
        memory = memory_estimates(self.config, self.inputs)
        self.assertEqual(memory["engine_count"], 3)
        self.assertEqual(memory["engine_count"] * memory["requests_per_engine"], 192)
        self.assertEqual(self.config["miles_args"]["global_batch_size"], 2048)
        self.assertGreater(memory["engine_remaining_planning_headroom_gib"], 0)

    def test_scenarios_retain_cap_arithmetic_and_mark_assumed_rates(self):
        row = next(
            row
            for row in compute_scenarios(self.config, self.inputs)
            if row["output_tokens_per_sample"] == 90000 and row["global_batch"] == 2048
        )
        self.assertEqual(row["output_tokens_per_update"], 184320000)
        self.assertEqual(row["rollout_hours_at_aggregate_1000_tokens_s"], 51.2)


if __name__ == "__main__":
    unittest.main()
