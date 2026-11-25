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
            complete(root, screen=screen, excluded=excluded)
            cases = json.loads((packet / "packet.json").read_text())["cases"]
            selected = json.loads((packet / "selection-provenance-not-for-labeling.json").read_text())
            self.assertEqual(len(cases), 40)
            self.assertEqual(len({row["question_key"] for row in selected}), 6)
            self.assertIn(rows[31]["judge_identity"], {row["judge_identity"] for row in selected})
            self.assertFalse(
                {row["judge_identity"] for row in rows[:31]} & {row["judge_identity"] for row in selected}
            )
            self.assertEqual(
                json.loads((packet / "packet-freeze.json").read_text())["packet_sha256"],
                digest(packet / "packet.json"),
            )


if __name__ == "__main__":
    unittest.main()
