import gzip
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.tcs_async_rl.diagnose_rl import (
    per_question_steps,
    question_comparisons,
    read_samples,
    validate_group_metrics,
)
from experiments.tcs_async_rl.reward_hook import question_key


class MultiGroupAuditTests(unittest.TestCase):
    def test_four_prompt_groups_remain_distinct_and_match_the_global_native_mean(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            trace_dir, output = root / "traces", root / "audit"
            (trace_dir / "rollout_data").mkdir(parents=True)
            output.mkdir()
            questions = [
                {
                    "problem": f"Q{i}",
                    "answer": "reference",
                    "rubric": "1. [7 pt] proof",
                    "paper_id": f"paper{i}",
                    "prompt_type": "proof",
                    "difficulty": "easy",
                }
                for i in range(4)
            ]
            (root / "input.json").write_text(json.dumps({"questions": questions}))
            (root / "resolved-config.json").write_text(
                json.dumps({"args": {"global_batch_size": 128, "n_samples_per_prompt": 32, "seed": 20261124}})
            )
            rows = [
                {
                    "index": g * 32 + i,
                    "group_index": g,
                    "optimizer_rollout_id": 109,
                    "metadata": {"question_key": question_key(questions[g])},
                    "judge_identity": str(g * 32 + i),
                    "reward": {"reward": g + i % 2},
                    "status": "completed",
                    "response": "thinking</think>answer",
                    "final_answer": "answer",
                }
                for g in range(4)
                for i in range(32)
            ]
            source = trace_dir / "rollout_data/rollout-109.jsonl.gz"

            def write(values):
                with gzip.open(source, "wt") as stream:
                    for row in values:
                        stream.write(json.dumps(row) + "\n")
                (root / "archive-index.json").write_text(
                    json.dumps(
                        {
                            "rollout_data/rollout-109.jsonl.gz": {
                                "sha256": hashlib.sha256(source.read_bytes()).hexdigest()
                            }
                        }
                    )
                )

            write(rows)
            with patch(
                "experiments.tcs_async_rl.diagnose_rl.judge_receipts",
                side_effect=lambda directory, question, sample: {
                    "order_scores": [sample["reward"]["reward"]],
                    "reference_scores": [],
                    "candidate_decisions": [[]],
                },
            ), patch(
                "experiments.tcs_async_rl.diagnose_rl.trace_summary",
                side_effect=lambda sample: {"response_tokens": 10 + sample["group_index"]},
            ):
                samples, groups = read_samples(root, trace_dir, output, optimizer_steps=[109])
                self.assertEqual(len(samples), 128)
                self.assertEqual([g["samples"] for g in groups], [32] * 4)
                self.assertEqual([g["mean_reward"] for g in groups], [0.5, 1.5, 2.5, 3.5])
                self.assertEqual(len(list(output.glob("example-*.json"))), 4)
                self.assertTrue(all(g["mean_order_gap"] is None for g in groups))
                native = [{"optimizer_step": 109, "rollout/episode_raw_reward": 2, "rollout/response_lengths": 11.5}]
                validate_group_metrics(groups, native)
                with self.assertRaises(ValueError):
                    validate_group_metrics(groups, [{**native[0], "rollout/episode_raw_reward": 0.5}])
                write(rows[:-1])
                with self.assertRaises(AssertionError):
                    read_samples(root, trace_dir, output, optimizer_steps=[109])

    def test_repeated_groups_in_one_update_are_not_mistaken_for_learning(self):
        base = {
            "question_key": "q",
            "paper_id": "paper",
            "samples": 32,
            "reward_std": 0,
            "cap_rate": 0,
            "mean_tokens": 10,
            "full_credit_rate": 0,
            "zero_rate": 0,
        }
        groups = [
            {**base, "optimizer_step": 109, "group_index": 0, "mean_reward": 1},
            {**base, "optimizer_step": 109, "group_index": 1, "mean_reward": 3},
        ]
        merged = per_question_steps(groups)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["samples"], 64)
        self.assertEqual(merged[0]["mean_reward"], 2)
        self.assertAlmostEqual(merged[0]["reward_std"], math.sqrt(64 / 63))
        self.assertEqual(question_comparisons(groups), [])
        pairs = question_comparisons(groups + [{**base, "optimizer_step": 110, "group_index": 2, "mean_reward": 4}])
        self.assertEqual(pairs[0]["first_reward"], 2)
        self.assertEqual(pairs[0]["difference"], 2)


if __name__ == "__main__":
    unittest.main()
