import unittest

from experiments.tcs_async_rl.vllm_group import completion_output, completion_payload


class CompletionTransportTests(unittest.TestCase):
    def test_frozen_token_ids_and_sampling_controls_reach_the_endpoint(self):
        sampling = {"max_new_tokens": 90000, "temperature": 1.0, "top_p": 1.0, "top_k": -1, "sampling_seed": 20261118}
        payload = completion_payload([1, 2, 3], sampling, "fixed-model")
        self.assertEqual(payload["prompt"], [1, 2, 3])
        self.assertEqual(payload["max_tokens"], 90000)
        self.assertEqual(payload["seed"], 20261118)
        self.assertFalse(payload["add_special_tokens"])
        self.assertFalse(payload["skip_special_tokens"])

    def test_server_usage_and_raw_thinking_are_preserved(self):
        reply = {
            "choices": [{"text": "reasoning</think>answer<|im_end|>", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 374, "completion_tokens": 20},
        }
        result = completion_output(reply)
        self.assertEqual(result["text"], reply["choices"][0]["text"])
        self.assertEqual(result["meta_info"]["prompt_tokens"], 374)
        self.assertEqual(result["raw_api_response"], reply)
        reply["choices"][0]["finish_reason"] = "error"
        with self.assertRaises(ValueError):
            completion_output(reply)


if __name__ == "__main__":
    unittest.main()
