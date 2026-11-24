import copy
import unittest

from experiments.tcs_async_rl.compare_screens import compare


def screen(checkpoint, score):
    return {
        "checkpoint_rollout_id": checkpoint,
        "optimizer_updates": 0,
        "grade_method": "reference_guided",
        "judge_reasoning_effort": "high",
        "dataset_sha256": "same-data",
        "judge_prompt_sha256": "same-prompt",
        "judge_config_sha256": "same-config",
        "sampling": {"seed": 20261124, "max_new_tokens": 90000},
        "answers": 384,
        "questions": [
            {
                "question_key": f"q{index}",
                "paper_id": f"p{index}",
                "difficulty": "easy" if index < 6 else "medium",
                "answers": 32,
                "reward_counts": {str(score): 32},
                "mean_reward": score,
                "full_credit": 32 if score == 7 else 0,
                "capped": 0,
                "mean_response_tokens": 100,
            }
            for index in range(12)
        ],
    }


class ScreenComparisonTests(unittest.TestCase):
    def test_pairs_by_question_identity_and_keeps_difficulties_separate(self):
        before, after = screen(108, 1), screen(120, 2)
        after["questions"].reverse()
        result = compare(before, after)
        self.assertTrue(result["paired"])
        self.assertEqual([row["change_mean_reward"] for row in result["questions"]], [1] * 12)
        self.assertEqual(result["by_difficulty"]["medium"]["answers_per_checkpoint"], 192)
        self.assertIsNone(compare(before)["by_difficulty"]["easy"]["after_mean_reward"])

    def test_rejects_a_changed_grader_or_wrong_checkpoint_instead_of_calling_it_learning(self):
        before = screen(108, 1)
        for key in ("judge_prompt_sha256", "judge_config_sha256", "dataset_sha256", "sampling"):
            after = screen(120, 2)
            after[key] = "different"
            with self.assertRaises(ValueError):
                compare(before, after)
        with self.assertRaises(ValueError):
            compare(before, screen(108, 2))

    def test_rejects_incomplete_or_duplicated_question_groups_and_inconsistent_scores(self):
        original = screen(108, 1)
        broken = []
        value = copy.deepcopy(original)
        value["questions"][0]["reward_counts"] = {"1": 31}
        broken.append(value)
        value = copy.deepcopy(original)
        value["questions"][0]["mean_reward"] = 7
        broken.append(value)
        value = copy.deepcopy(original)
        value["questions"][0] = value["questions"][1]
        broken.append(value)
        for value in broken:
            with self.assertRaises(ValueError):
                compare(value)


if __name__ == "__main__":
    unittest.main()
