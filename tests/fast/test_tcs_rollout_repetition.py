import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.tcs_async_rl.rollout_repetition import analyze, repeated_windows


class RolloutRepetitionTests(unittest.TestCase):
    def test_exact_windows_and_short_answer_denominator(self):
        self.assertEqual(repeated_windows("a b c d", 3)["repeated_fraction"], 0)
        self.assertEqual(repeated_windows("x x x x", 3)["repeated_fraction"], 0.5)
        self.assertIsNone(repeated_windows("short", 3)["repeated_fraction"])
        with self.assertRaises(ValueError):
            repeated_windows("a b", 0)

    def test_archived_thinking_and_final_repetition_remain_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "rollout_data/rollout-12.jsonl.gz"
            path.parent.mkdir()
            samples = [
                ("completed", "a b c d", "a b c", 1),
                ("truncated", "x x x x", None, 0),
                ("truncated", "p q p q p", "z z z z", 7),
            ]
            with gzip.open(path, "wt") as stream:
                for number, (status, response, final, reward) in enumerate(samples):
                    stream.write(
                        json.dumps(
                            {
                                "optimizer_rollout_id": 12,
                                "index": number,
                                "metadata": {"question_key": "question"},
                                "status": status,
                                "response": response,
                                "final_answer": final,
                                "response_length": 4,
                                "reward": {"reward": reward},
                            }
                        )
                        + "\n"
                    )
            entry = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
            (root / "archive-index.json").write_text(json.dumps({str(path.relative_to(root)): entry}))
            result = analyze(root, root, [12], window_words=3)
            groups = result["by_completion_outcome"]
            self.assertEqual(result["answers"], 3)
            self.assertEqual(groups["capped_empty_final"]["full_response"]["median_repeated_fraction"], 0.5)
            self.assertEqual(groups["capped_empty_final"]["final_answer"]["measured_answers"], 0)
            self.assertAlmostEqual(groups["capped_nonempty_final"]["full_response"]["median_repeated_fraction"], 1 / 3)
            self.assertEqual(groups["capped_nonempty_final"]["final_answer"]["median_repeated_fraction"], 0.5)
            self.assertEqual(groups["capped_nonempty_final"]["full_credit"], 1)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "immutable archived trace"):
                analyze(root, root, [12], window_words=3)
