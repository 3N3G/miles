import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.tcs_async_rl.reward_hook import reward
from experiments.tcs_async_rl.rl_artifacts import portable_sample, save_generated_sample


class Array:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class Mask:
    def _as_tensors(self):
        return Array([2, 4, 3]), Array([0, 2, 3])


def sample_dict():
    return {
        "index": 7,
        "group_index": 1,
        "tokens": [1, 2, 3],
        "response": "thinking</think>answer",
        "response_length": 2,
        "status": "completed",
        "metadata": {"question_key": "question"},
        "rollout_log_probs": [-0.25, -0.5],
        "rollout_sampling_mask": Mask(),
    }


class ArtifactTests(unittest.TestCase):
    def test_replay_inputs_survive_portable_export_and_identical_raw_data_is_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            pattern = str(Path(directory) / "rollout-{rollout_id}.pt")
            first = save_generated_sample(pattern, sample_dict())
            second = save_generated_sample(pattern, sample_dict())
            self.assertEqual(first, second)
            saved = json.loads(first.read_text())
            self.assertEqual(saved["tokens"], [1, 2, 3])
            self.assertEqual(saved["rollout_log_probs"], [-0.25, -0.5])
            self.assertEqual(saved["rollout_sampling_mask"]["ids"], [2, 4, 3])
            self.assertEqual(saved["rollout_sampling_mask"]["offsets"], [0, 2, 3])
            self.assertEqual(saved["final_answer"], "answer")

    def test_recorded_empty_model_answer_links_to_its_actual_empty_judge_receipt(self):
        saved = portable_sample({**sample_dict(), "response": "Wait.", "status": "truncated"})
        self.assertIsNone(saved["final_answer"])
        self.assertEqual(saved["judge_identity"], "online-e3b0c44298fc1c149afbf4c8")


class HookFailureEvidenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_failure_preserves_model_output_and_does_not_create_a_zero_reward(self):
        class Response:
            status = 502

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        class Session(Response):
            def __init__(self, **kwargs):
                pass

            def post(self, *args, **kwargs):
                return Response()

        fake_http = SimpleNamespace(ClientSession=Session, ClientTimeout=lambda **kwargs: None)
        sample = SimpleNamespace(
            response="thinking</think>answer",
            status=SimpleNamespace(value="completed"),
            metadata={"question_key": "question"},
            to_dict=sample_dict,
        )
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                rm_url="http://127.0.0.1:18491/score",
                save_debug_rollout_data=str(Path(directory) / "rollout-{rollout_id}.pt"),
            )
            with patch.dict(sys.modules, {"aiohttp": fake_http}):
                with self.assertRaisesRegex(RuntimeError, "HTTP 502"):
                    await reward(args, sample)
            files = list((Path(directory) / "completed_samples").glob("*.generated.json"))
            self.assertEqual(len(files), 1)
            self.assertEqual(json.loads(files[0].read_text())["tokens"], [1, 2, 3])
            outcomes = list(files[0].parent.glob("*.judge-*.json"))
            self.assertEqual(len(outcomes), 1)
            outcome = json.loads(outcomes[0].read_text())
            self.assertIsNone(outcome["result"])
            self.assertEqual(outcome["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
