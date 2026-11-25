import math
import unittest

from experiments.tcs_async_rl.rollout_support import aggregate, position_counts


class RolloutSupportTests(unittest.TestCase):
    def test_position_fractions_pool_tokens_instead_of_averaging_different_answer_lengths(self):
        longer = {
            "response_length": 4,
            "rollout_sampling_mask": {"ids": [7, 7, 9, 7, 7, 9], "offsets": [0, 1, 3, 4, 6]},
            "rollout_log_probs": [0, -math.log(2), 0, -math.log(2)],
        }
        shorter = {
            "response_length": 2,
            "rollout_sampling_mask": {"ids": [7, 7], "offsets": [0, 1, 2]},
            "rollout_log_probs": [0, 0],
        }
        rows = [
            {"outcome": "completed", "positions": position_counts(sample, 9, 2)}
            for sample in (longer, shorter)
        ]
        result = aggregate(rows, 2)["completed"]
        self.assertEqual(result["answers"], 2)
        for bucket in result["positions"]:
            self.assertEqual(bucket["tokens"], 3)
            self.assertAlmostEqual(bucket["single_choice_fraction"], 2 / 3)
            self.assertAlmostEqual(bucket["end_thinking_admitted_fraction"], 1 / 3)
            self.assertAlmostEqual(bucket["mean_sampled_surprisal_nats"], math.log(2) / 3)


if __name__ == "__main__":
    unittest.main()
