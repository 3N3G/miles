import ast
import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from experiments.tcs_async_rl.checkpoint_screen import verify_publication
from experiments.tcs_async_rl.rl_inputs import question_set_key, read_bundle
from experiments.tcs_async_rl.screen_inputs import select_questions


class CheckpointScreenTests(unittest.TestCase):
    def test_fresh_phase_does_not_read_old_queues_but_default_restore_still_requires_them(self):
        source = ast.parse(Path("miles/ray/rollout/rollout_executor.py").read_text())
        method = next(
            node for node in ast.walk(source) if isinstance(node, ast.AsyncFunctionDef) and node.name == "load"
        )
        with tempfile.TemporaryDirectory() as name:
            old_directory = Path(name) / "old-rollout"
            locate = Mock(return_value=old_directory)
            namespace = {"logger": Mock(), "compute_rollout_checkpoint_dir": locate}
            exec(
                compile(
                    ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                    "actual-rollout-restore",
                    "exec",
                ),
                namespace,
            )
            actor = SimpleNamespace(args=SimpleNamespace(reset_rollout_state=True, load="protected-source"))
            asyncio.run(namespace["load"](actor, 108))
            locate.assert_not_called()
            actor.args.reset_rollout_state = False
            with self.assertRaises(AssertionError):
                asyncio.run(namespace["load"](actor, 108))
            locate.assert_called_once()

    def test_selection_is_seeded_and_uses_both_strata_and_distinct_papers(self):
        rows = [
            {"difficulty": level, "cohort": "SODA_2023", "paper_id": f"{level}-{index}", "node_id": index}
            for level in ("easy", "medium")
            for index in range(20)
        ]
        selected = select_questions(rows)
        self.assertEqual(selected, select_questions(rows))
        self.assertEqual([row["difficulty"] for row in selected], ["easy"] * 6 + ["medium"] * 6)
        self.assertEqual(len({row["paper_id"] for row in selected}), 12)
        self.assertNotEqual(selected, select_questions(rows, seed=1))

    def test_screen_bundle_requires_the_explicit_frozen_contract(self):
        row = {
            "problem": "q",
            "answer": "a",
            "rubric": "1. [7 pts] proof",
            "difficulty": "medium",
            "cohort": "SODA_2023",
        }
        training = {"data": {"revision": "revision", "eligible_configs": ["SODA_2023"]}}
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "bundle.json"
            path.write_text(
                json.dumps(
                    {"questions": [row], "dataset_revision": "revision", "question_set_key": question_set_key([row])}
                )
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                read_bundle(path, expected_sha256=digest, training=training)
            rows, _ = read_bundle(
                path, expected_sha256=digest, training=training, contract={"questions": 1, "difficulties": ["medium"]}
            )
            self.assertEqual(rows, [row])

    def test_screen_rejects_weights_that_differ_from_the_saved_publication(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source, current = root / "source.jsonl", root / "current.jsonl"
            event = {
                "type": "inference_engine_weight_checksum",
                "rollout_id": 108,
                "engine_checksums": [{"weight": 1}] * 3,
            }
            source.write_text(json.dumps(event) + "\n")
            current.write_text(json.dumps(event) + "\n")
            self.assertTrue(verify_publication(source, current, rollout_id=108)["matches_source"])
            event["engine_checksums"] = [{"weight": 2}] * 3
            current.write_text(json.dumps(event) + "\n")
            with self.assertRaisesRegex(ValueError, "differ"):
                verify_publication(source, current, rollout_id=108)


if __name__ == "__main__":
    unittest.main()
