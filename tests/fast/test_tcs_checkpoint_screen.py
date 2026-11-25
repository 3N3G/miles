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
from experiments.tcs_async_rl.regrade_live_screen import fingerprint, validation_receipts
from experiments.tcs_async_rl.rl_inputs import question_set_key, read_bundle
from experiments.tcs_async_rl.rl_smoke import validate_judge_qualification
from experiments.tcs_async_rl.screen_inputs import select_questions


class CheckpointScreenTests(unittest.TestCase):
    def test_baseline_reuse_rejects_changed_or_unfinished_requests(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            path = root / "original-case-order0.json"
            payload = {"model": "judge", "messages": [{"content": "frozen input"}]}
            receipt = {
                "payload": payload,
                "request_sha256": fingerprint(payload),
                "http_status": 200,
                "swapped": False,
                "body": {"choices": [{"finish_reason": "stop"}]},
            }
            path.write_text(json.dumps(receipt))
            (root / "verdicts.json").write_text(
                json.dumps({"case-01": {"reference_v5": [{"receipt": str(path)}]}})
            )
            self.assertEqual(validation_receipts(root), {fingerprint(payload): path})
            receipt["payload"]["messages"][0]["content"] = "different input"
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "changed request"):
                validation_receipts(root)
            receipt["request_sha256"] = fingerprint(receipt["payload"])
            receipt["body"]["choices"][0]["finish_reason"] = "length"
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                validation_receipts(root)

    def test_failed_observed_gate_needs_exact_scoped_user_acceptance(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            values = {
                "runtime.json": {
                    "grading_method": "reference_guided",
                    "training_config": "training.json",
                    "prompt_file": "prompt.txt",
                },
                "training.json": {"judge": {"primary_model": "gpt-6-luna", "reasoning_effort": "high"}},
                "selection.json": {"selected_method": "reference_guided"},
                "comparison.json": {"reference_guided": {"passes_observed_gate": False}},
                "controls.json": {"failures": []},
            }
            for file, value in values.items():
                (root / file).write_text(json.dumps(value))
            (root / "prompt.txt").write_text("fixed prompt")

            def digest(file):
                return hashlib.sha256((root / file).read_bytes()).hexdigest()

            q = {
                key: {"file": key + ".json", "sha256": digest(key + ".json")}
                for key in ("selection", "comparison", "controls")
            }
            q.update(
                runtime_config_sha256=digest("runtime.json"),
                training_config_sha256=digest("training.json"),
                prompt_sha256=digest("prompt.txt"),
            )
            config = {
                "judge_config": "runtime.json",
                "training_config": "training.json",
                "scope": {"prompt_count": 236},
                "judge_qualification": q,
            }
            with self.assertRaises(ValueError):
                validate_judge_qualification(config, root)
            acceptance = {
                "decision": "approved_launch_with_observed_judge_limitations",
                "prompt_sha256": digest("prompt.txt"),
                "judge_model": "gpt-6-luna",
                "reasoning_effort": "high",
                "questions": 236,
                "user_statement": "Then launch I give you permission",
            }
            (root / "acceptance.json").write_text(json.dumps(acceptance))
            q["user_acceptance"] = {"file": "acceptance.json", "sha256": digest("acceptance.json")}
            self.assertEqual(validate_judge_qualification(config, root), root / "selection.json")
            config["scope"]["prompt_count"] = 20
            with self.assertRaisesRegex(ValueError, "scope"):
                validate_judge_qualification(config, root)
            config["scope"]["prompt_count"] = 236
            (root / "prompt.txt").write_text("changed prompt")
            with self.assertRaises(ValueError):
                validate_judge_qualification(config, root)

    def test_fresh_phase_does_not_read_old_queues_but_default_restore_still_requires_them(self):
        source = ast.parse(Path("miles/ray/rollout/rollout_executor.py").read_text())
        method = next(
            node
            for node in ast.walk(source)
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "load"
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
                    {
                        "questions": [row],
                        "dataset_revision": "revision",
                        "question_set_key": question_set_key([row]),
                    }
                )
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                read_bundle(path, expected_sha256=digest, training=training)
            rows, _ = read_bundle(
                path,
                expected_sha256=digest,
                training=training,
                contract={"questions": 1, "difficulties": ["medium"]},
            )
            self.assertEqual(rows, [row])

    def test_screen_rejects_weights_that_differ_from_the_saved_publication(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source, current = root / "source.jsonl", root / "current.jsonl"
            restored = root / "restored.jsonl"
            restored.write_text("")
            event = {
                "type": "inference_engine_weight_checksum",
                "rollout_id": 108,
                "engine_checksums": [{"weight": 1}] * 3,
            }
            source.write_text(json.dumps(event) + "\n")
            current.write_text(json.dumps(event) + "\n")
            self.assertTrue(
                verify_publication(
                    source, current, rollout_id=108, restored_events=restored, expected_publications=1
                )["matches_source"]
            )
            event["engine_checksums"] = [{"weight": 2}] * 3
            current.write_text(json.dumps(event) + "\n")
            with self.assertRaisesRegex(ValueError, "differ"):
                verify_publication(
                    source, current, rollout_id=108, restored_events=restored, expected_publications=1
                )

    def test_resume_verifies_both_transfers_and_excludes_inherited_publications(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source, current, restored = (
                root / name for name in ("source.jsonl", "current.jsonl", "restored.jsonl")
            )
            event = {
                "type": "inference_engine_weight_checksum",
                "rollout_id": 108,
                "engine_checksums": [{"w": 1}] * 3,
            }
            inherited = {**event, "rollout_id": 107}
            prefix = json.dumps(inherited) + "\n"
            restored.write_text(prefix)
            source.write_text(prefix + json.dumps(event) + "\n")
            current.write_text(prefix + (json.dumps(event) + "\n") * 2)
            options = {"rollout_id": 108, "restored_events": restored, "expected_publications": 2}
            result = verify_publication(source, current, **options)
            self.assertEqual(result["new_publications_verified"], 2)
            for count in (1, 3):
                current.write_text(prefix + (json.dumps(event) + "\n") * count)
                with self.assertRaisesRegex(ValueError, "count"):
                    verify_publication(source, current, **options)
            current.write_text(
                prefix
                + json.dumps(event)
                + "\n"
                + json.dumps({**event, "engine_checksums": [{"w": 2}] * 3})
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "differ"):
                verify_publication(source, current, **options)
            current.write_text((json.dumps(event) + "\n") * 2)
            with self.assertRaisesRegex(ValueError, "history"):
                verify_publication(source, current, **options)


if __name__ == "__main__":
    unittest.main()
