import unittest

from experiments.tcs_async_rl.policy_reference import compare_logprobs


class TestPolicyReference(unittest.TestCase):
    def test_near_tie_can_change_greedy_token_without_large_numerical_error(self):
        result = compare_logprobs(
            {10: -0.70, 11: -0.71}, {10: -0.72, 11: -0.70}, chosen=10, reference_best=11, tolerance=0.2
        )
        self.assertTrue(result["passed"])
        self.assertFalse(result["same_greedy_token"])
        self.assertAlmostEqual(result["reference_greedy_gap_nats"], 0.02)

    def test_large_error_or_nonfinite_values_do_not_pass(self):
        result = compare_logprobs({10: -0.5}, {10: -1.0}, chosen=10, reference_best=10, tolerance=0.2)
        self.assertFalse(result["passed"])
        with self.assertRaises(ValueError):
            compare_logprobs({10: float("nan")}, {10: -1.0}, chosen=10, reference_best=10, tolerance=0.2)


if __name__ == "__main__":
    unittest.main()
