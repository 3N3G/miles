import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from experiments.tcs_async_rl.reward_hook import prepare_smoke_row, question_key, scoring_request, validate_reward
from experiments.tcs_async_rl.reward_service import Scorer


class RewardHookTests(unittest.TestCase):
    def test_only_a_recorded_model_length_limit_allows_an_empty_final(self):
        sample = SimpleNamespace(
            status=SimpleNamespace(value="truncated"),
            response="Wait. Wait.",
            metadata={"question_key": "question"},
        )
        self.assertTrue(scoring_request(sample)["generator_truncated"])
        sample.status.value = "aborted"
        with self.assertRaisesRegex(ValueError, "token limit"):
            scoring_request(sample)

    def test_actor_prompt_contains_no_solution_or_rubric(self):
        row = {
            "problem": "question",
            "answer": "solution",
            "rubric": "criterion",
            "cohort": "SODA_2023",
            "paper_id": "paper",
            "node_id": "node",
        }
        prepared = prepare_smoke_row(row)
        self.assertEqual(prepared["prompt"], "question")
        self.assertNotIn("rubric", prepared)
        self.assertEqual(prepared["metadata"]["question_key"], question_key(row))
        self.assertNotEqual(question_key(row), question_key({**row, "rubric": "changed"}))

    def test_invalid_judge_result_cannot_become_zero(self):
        for value in (None, True, "0", -1, 8, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                validate_reward({"reward": value, "judge_order_gap": 0})
        with self.assertRaises(ValueError):
            validate_reward({"reward": 1, "judge_order_gap": float("nan")})
        self.assertEqual(validate_reward({"reward": 0, "judge_order_gap": 1}), {"reward": 0.0, "judge_order_gap": 1.0})


class RewardServiceTests(unittest.IsolatedAsyncioTestCase):
    def scorer(self, directory):
        # Avoid credentials and paid calls. Exercise the live scoring path with a fake judge.
        scorer = Scorer.__new__(Scorer)
        scorer.question_key = "approved-question"
        scorer.row = {"problem": "question"}
        scorer.judge = scorer.config = {}
        scorer.prompt = "judge"
        scorer.key = "unused-test-placeholder"
        scorer.budget = None
        scorer.directory = Path(directory)
        scorer.semaphore = asyncio.Semaphore(4)
        scorer.in_flight = {}
        return scorer

    async def test_order_average_and_concurrent_request_deduplication(self):
        async def grade(**kwargs):
            await asyncio.sleep(0)
            return {"candidate": {"score": 3 if kwargs["swapped"] else 2}}

        with tempfile.TemporaryDirectory() as directory:
            scorer = self.scorer(directory)
            request = {"question_key": scorer.question_key, "response": "thinking</think>final answer"}
            with patch("experiments.tcs_async_rl.reward_service.grade_one", side_effect=grade) as mocked:
                first, duplicate = await asyncio.gather(scorer.score(request), scorer.score(request))
            self.assertEqual(first, duplicate)
            self.assertEqual(first["reward"], 2.5)
            self.assertEqual(first["judge_order_gap"], 1)
            self.assertEqual(mocked.call_count, 2)

    async def test_incomplete_and_wrong_question_never_call_judge(self):
        with tempfile.TemporaryDirectory() as directory:
            scorer = self.scorer(directory)
            with patch("experiments.tcs_async_rl.reward_service.grade_one", new_callable=AsyncMock) as mocked:
                for request in (
                    {"question_key": "other", "response": "</think>answer"},
                    {"question_key": scorer.question_key, "response": "unfinished thinking"},
                    [],
                ):
                    with self.assertRaises(ValueError):
                        await scorer.score(request)
                mocked.assert_not_called()

    async def test_failed_judge_is_retryable_and_never_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            scorer = self.scorer(directory)
            request = {"question_key": scorer.question_key, "response": "</think>answer"}
            with patch(
                "experiments.tcs_async_rl.reward_service.grade_one", side_effect=RuntimeError("judge unavailable")
            ):
                with self.assertRaises(RuntimeError):
                    await scorer.score(request)
            self.assertFalse(scorer.in_flight)

    async def test_model_length_limit_uses_the_real_empty_answer_judgments(self):
        async def grade(**kwargs):
            self.assertEqual(kwargs["candidate"], "")
            return {"candidate": {"score": 0}}

        with tempfile.TemporaryDirectory() as directory:
            scorer = self.scorer(directory)
            request = {
                "question_key": scorer.question_key,
                "response": "Wait. Wait.",
                "generator_truncated": True,
            }
            with patch("experiments.tcs_async_rl.reward_service.grade_one", side_effect=grade) as mocked:
                result = await scorer.score(request)
            self.assertEqual(mocked.call_count, 2)
            self.assertEqual(result["reward"], 0)
            self.assertTrue(result["generator_truncated"])
            scorer.in_flight.clear()
            with patch("experiments.tcs_async_rl.reward_service.grade_one", return_value={"candidate": {"score": 1}}):
                with self.assertRaisesRegex(ValueError, "Empty-final rubric control failed"):
                    await scorer.score(request)
            with patch(
                "experiments.tcs_async_rl.reward_service.grade_one", side_effect=RuntimeError("provider failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "provider failed"):
                    await scorer.score(request)


if __name__ == "__main__":
    unittest.main()
