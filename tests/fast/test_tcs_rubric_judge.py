import json
import tempfile
import unittest
from pathlib import Path

from experiments.tcs_async_rl.grade_group import Budget, reserve_cost, usage_cost
from experiments.tcs_async_rl.judge_client import load_credentials
from experiments.tcs_async_rl.judge_comparison import comparison_metrics
from experiments.tcs_async_rl.judge_comparison_checks import clustered_intervals
from experiments.tcs_async_rl.rubric_judge import messages, parse_grade, rubric_items


class RubricJudgeTests(unittest.TestCase):
    def test_supplied_weights_and_zero_credit_examples(self):
        rubric = "Total: 7 points\n1. [2 pt] A\n2. [5 pt] B\nZero-credit items:\n1. A mistake."
        items = rubric_items(rubric)
        self.assertEqual(items, [(1, 2), (2, 5)])
        verdict = {
            name: [
                {"item": 1, "satisfied": True, "reason": "proved"},
                {"item": 2, "satisfied": False, "reason": "missing"},
            ]
            for name in ("A", "B")
        }
        self.assertEqual(parse_grade(json.dumps(verdict), items)["A"]["score"], 2)

    def test_invalid_verdict_is_not_a_zero_grade(self):
        for content in ("{}", '{"A":[],"B":[]}', "not JSON"):
            with self.assertRaises(ValueError):
                parse_grade(content, [(1, 7)])

    def test_answer_order_reverses_without_source_labels(self):
        row = {"problem": "problem", "rubric": "rubric", "answer": "reference text"}
        first = json.loads(messages(row, "candidate text", swapped=False, system_prompt="judge")[1]["content"])
        second = json.loads(messages(row, "candidate text", swapped=True, system_prompt="judge")[1]["content"])
        self.assertEqual(first["answers"]["A"], second["answers"]["B"])
        self.assertEqual(set(first), {"problem", "rubric", "answers"})

    def test_reference_guided_request_identifies_roles_and_rejects_swapping(self):
        row = {"problem": "problem", "rubric": "rubric", "answer": "reference text"}
        request = messages(
            row, "candidate text", swapped=False, system_prompt="judge", grading_method="reference_guided"
        )
        body = json.loads(request[1]["content"])
        self.assertEqual(body["reference_solution"], "reference text")
        self.assertEqual(body["candidate_solution"], "candidate text")
        self.assertNotIn("answers", body)
        with self.assertRaises(ValueError):
            messages(row, "candidate text", swapped=True, system_prompt="judge", grading_method="reference_guided")

    def test_single_candidate_parser_keeps_binary_weighted_scoring(self):
        items = [(1, 2), (2, 5)]
        decisions = [
            {"item": 1, "satisfied": True, "reason": "established"},
            {"item": 2, "satisfied": False, "reason": "missing"},
        ]
        parsed = parse_grade(json.dumps({"candidate": decisions}), items, answer_labels=("candidate",))
        self.assertEqual(parsed["candidate"]["score"], 2)
        with self.assertRaises(ValueError):
            parse_grade(json.dumps({"A": decisions, "B": decisions}), items, answer_labels=("candidate",))

    def test_comparison_counts_one_answer_and_averages_presentation_scores(self):
        case = {"case": "case-01", "question": {"problem": "P", "answer": "R", "rubric": "1. [2 pt] X\n2. [5 pt] Y"}}
        correct = [{"item": 1, "satisfied": True}, {"item": 2, "satisfied": False}]
        incorrect = [{"item": 1, "satisfied": False}, {"item": 2, "satisfied": False}]
        labels = {"cases": [{"case": "case-01", "candidate_items": correct}]}
        results = {
            "case-01": {
                "anonymous_pair": [
                    {"candidate": {"score": 2, "items": correct}},
                    {"candidate": {"score": 0, "items": incorrect}},
                ],
                "reference_guided": [{"candidate": {"score": 2, "items": correct}}],
            }
        }
        result = comparison_metrics([case], labels, results)
        self.assertEqual(result["anonymous_pair"]["answers"], 1)
        self.assertEqual(result["anonymous_pair"]["cases"][0]["reward"], 1)
        self.assertEqual(result["anonymous_pair"]["case_balanced_item_agreement"], 0.75)
        self.assertFalse(result["anonymous_pair"]["passes_observed_gate"])
        self.assertTrue(result["reference_guided"]["passes_observed_gate"])

    def test_budget_reservations_survive_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"usd_cap": 1}
            budget = Budget(Path(directory), config)
            budget.reserve(0.8, "first")
            with self.assertRaisesRegex(RuntimeError, "cap"):
                Budget(Path(directory), config).reserve(0.3, "second")
            budget.settle(0.8, 0.2, "first")
            self.assertAlmostEqual(Budget(Path(directory), config).total, 0.2)

    def test_bootstrap_keeps_same_question_answers_in_one_cluster(self):
        rows = [
            {
                "question_key": str(value),
                "exact_score_match": bool(value),
                "item_agreement": value,
                "absolute_score_error": 1 - value,
            }
            for value in (0, 1)
            for _ in range(8)
        ]
        result = clustered_intervals({"paired": {"cases": rows}, "single": {"cases": rows}}, repetitions=500)
        self.assertEqual(result["question_clusters"], 2)
        self.assertEqual(result["methods"]["paired"]["item_agreement"], [0, 1])
        self.assertEqual(result["difference"]["intervals"]["item_agreement"], [0, 0])

    def test_price_arithmetic_and_conservative_reservation(self):
        config = json.loads(Path("experiments/tcs_async_rl/judge_group_v1.json").read_text())
        self.assertAlmostEqual(usage_cost({"prompt_tokens": 100000, "completion_tokens": 1000}, config), 0.0105)
        self.assertGreater(
            reserve_cost({"messages": [{"content": "a" * 10000}], "max_completion_tokens": 8192}, config), 0.004096
        )

    def test_parallel_budget_objects_share_the_spending_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Budget(Path(directory), {"usd_cap": 1})
            second = Budget(Path(directory), {"usd_cap": 1})
            first.reserve(0.8, "first")
            with self.assertRaisesRegex(RuntimeError, "cap"):
                second.reserve(0.3, "second")

    def test_local_loader_does_not_evaluate_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dummy-assignments"
            path.write_text('OPENAI_API_KEY="$(unsafe)"\n')
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                load_credentials(path)


if __name__ == "__main__":
    unittest.main()
