"""Dataset boundary, answer extraction, and incremental response persistence."""

import asyncio
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.tcs_async_rl.group_inference import final_answer, generate_group, load_input
from experiments.tcs_async_rl.watch_group import remote_state


class GroupInferenceTests(unittest.TestCase):
    def test_monitor_distinguishes_queued_from_completed(self):
        for status in ("PENDING", "COMPLETED"):
            result = subprocess.CompletedProcess([], 0, stdout=f"{status}|0:0|00:01:00|start|end\n")
            with patch("subprocess.run", return_value=result):
                self.assertEqual(remote_state("123")["state"], status)

    def test_monitor_rejects_a_non_numeric_job_id(self):
        with self.assertRaises(ValueError):
            remote_state("123;command")

    def test_thinking_only_is_not_a_zero_score(self):
        self.assertIsNone(final_answer("Let me reason for longer."))
        self.assertEqual(final_answer("thinking</think>\nA proof.<|im_end|>"), "A proof.")

    def test_private_input_identity_and_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            row = {"cohort": "FOCS_2026", "row": 1, "paper_id": "p", "node_id": "q"}
            path.write_text(json.dumps(row))
            config = {
                "input_file": str(path),
                "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "selection": dict(row),
            }
            with self.assertRaisesRegex(ValueError, "eligible"):
                load_input(config, {"data": {"eligible_configs": ["SODA_2023"]}})

    def test_changed_private_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "differs"):
                load_input({"input_file": str(path), "input_sha256": "wrong"}, {})

    def test_earlier_completions_survive_a_failed_peer(self):
        class FakeEngine:
            async def async_generate(self, *, input_ids, sampling_params):
                if sampling_params["sampling_seed"] == 11:
                    await asyncio.sleep(0.01)
                    raise RuntimeError("fake failure")
                return {
                    "text": "thinking</think>proof",
                    "meta_info": {"prompt_tokens": len(input_ids), "completion_tokens": 5},
                }

        training = {
            "miles_args": {
                "rollout_temperature": 1,
                "rollout_top_p": 1,
                "rollout_top_k": -1,
                "rollout_max_response_len": 90000,
                "n_samples_per_prompt": 2,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "fake failure"):
                asyncio.run(
                    generate_group(FakeEngine(), token_ids=[1, 2], training=training, seed=10, run_dir=Path(directory))
                )
            responses = [json.loads(line) for line in (Path(directory) / "responses.jsonl").read_text().splitlines()]
            self.assertEqual(len(responses), 1)
            self.assertEqual(responses[0]["sample_index"], 0)
            self.assertEqual(responses[0]["final_answer"], "proof")


if __name__ == "__main__":
    unittest.main()
