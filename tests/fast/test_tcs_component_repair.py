import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from experiments.tcs_async_rl.judge_comparison import digest, save
from experiments.tcs_async_rl.prepare_component_repair import complete
from experiments.tcs_async_rl.reward_hook import question_key


class ComponentPacketTests(unittest.TestCase):
    def test_two_reference_recipes_keep_the_same_agreement_gate_and_require_complete_labels(self):
        from experiments.tcs_async_rl.compare_reference_prompts import validate_labels
        from experiments.tcs_async_rl.judge_comparison import comparison_metrics

        question = {"problem": "problem", "answer": "reference", "rubric": "1. [7 pt] valid"}
        cases = [{"case": f"case-{i:02d}", "question": question, "candidate": str(i)} for i in range(40)]
        labels = {
            "cases": [
                {
                    "case": row["case"],
                    "candidate_sha256": hashlib.sha256(row["candidate"].encode()).hexdigest(),
                    "candidate_items": [{"item": 1, "satisfied": True, "reason": "valid"}],
                }
                for row in cases
            ]
        }
        validate_labels({"cases": cases}, labels)
        results = {
            row["case"]: {
                name: [{"candidate": {"score": 7 * ok, "items": [{"item": 1, "satisfied": ok}]}}]
                for name, ok in (("reference_v2", True), ("reference_v3", False))
            }
            for row in cases
        }
        metrics = comparison_metrics(cases, labels, results)
        self.assertTrue(metrics["reference_v2"]["passes_observed_gate"])
        self.assertFalse(metrics["reference_v3"]["passes_observed_gate"])
        for label in labels["cases"]:
            label["confidence"] = "medium"
            label["candidate_items"][0]["satisfied"] = False
        uncertain = comparison_metrics(cases, labels, results)["reference_v2"]
        self.assertEqual(uncertain["audit_false_full_credit"], 40)
        self.assertEqual(uncertain["clear_false_full_credit"], 0)
        self.assertFalse(uncertain["passes_observed_gate"])
        with self.assertRaises(ValueError):
            validate_labels({"cases": cases[:-1]}, labels)
        labels["cases"][0]["candidate_items"][0]["satisfied"] = None
        with self.assertRaises(ValueError):
            validate_labels({"cases": cases}, labels)

    def test_frozen_packet_covers_available_medium_questions_and_excludes_reviewed_answers(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            packet, screen = root / "packet", root / "screen"
            packet.mkdir()
            (screen / "rollout_data").mkdir(parents=True)
            question = {"problem": "easy", "answer": "reference", "rubric": "1. [7 pt] valid", "difficulty": "easy"}
            save(
                packet / "easy-packet.json",
                {
                    "cases": [
                        {"case": f"case-{i:02d}", "question": question, "candidate": str(i)} for i in range(1, 33)
                    ]
                },
            )
            save(packet / "easy-freeze.json", {"packet_sha256": digest(packet / "easy-packet.json")})
            save(packet / "selection-provenance-not-for-labeling.json", [])
            questions = [{**question, "problem": f"medium{i}", "difficulty": "medium"} for i in range(6)]
            questions.extend({**question, "problem": f"easy{i}"} for i in range(6))
            save(screen / "input.json", {"questions": questions})
            path = screen / "rollout_data/answers.jsonl.gz"
            rows = []
            for i, q in enumerate(questions):
                for index in range(32):
                    text = f"answer-{i}-{index}"
                    rows.append(
                        {
                            "index": i * 32 + index,
                            "metadata": {"question_key": question_key(q)},
                            "final_answer": text,
                            "judge_identity": hashlib.sha256(text.encode()).hexdigest(),
                        }
                    )
            with gzip.open(path, "wt") as stream:
                stream.write("\n".join(json.dumps(row) for row in rows))
            save(
                screen / "screen-complete.json",
                {"responses": str(path), "responses_sha256": digest(path), "answers": 384},
            )
            excluded = root / "excluded.json"
            key = question_key(questions[0])
            save(
                excluded,
                {
                    "question_key": key,
                    "audits": [{"identity": row["judge_identity"]} for row in rows[:31]],
                    "additional_7_point_case": {"identity": "other-reviewed"},
                },
            )
            previous = root / "previous"
            previous.mkdir()
            save(previous / "packet.json", {"cases": []})
            save(
                previous / "selection-provenance-not-for-labeling.json",
                [
                    {
                        "kind": "medium",
                        "question_key": question_key(questions[1]),
                        "judge_identity": row["judge_identity"],
                    }
                    for row in rows[32:64]
                ],
            )
            complete(root, screen=screen, excluded=excluded, previous=[previous])
            cases = json.loads((packet / "packet.json").read_text())["cases"]
            selected = json.loads((packet / "selection-provenance-not-for-labeling.json").read_text())
            self.assertEqual(len(cases), 40)
            self.assertEqual(len({row["question_key"] for row in selected}), 5)
            self.assertIn(rows[31]["judge_identity"], {row["judge_identity"] for row in selected})
            self.assertFalse(
                {row["judge_identity"] for row in rows[:31]} & {row["judge_identity"] for row in selected}
            )
            self.assertFalse(
                {row["judge_identity"] for row in rows[32:64]} & {row["judge_identity"] for row in selected}
            )
            self.assertEqual(
                json.loads((packet / "packet-freeze.json").read_text())["packet_sha256"],
                digest(packet / "packet.json"),
            )


if __name__ == "__main__":
    unittest.main()
