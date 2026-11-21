"""Per-sample Miles hook for the VM-owned TCS rubric service."""

import hashlib
import json
import math
import os

from experiments.tcs_async_rl.group_inference import final_answer
from experiments.tcs_async_rl.rl_artifacts import save_generated_sample, save_judge_outcome

REWARD_PROTOCOL_VERSION = 2


def question_key(row):
    values = {name: row[name] for name in ("problem", "answer", "rubric")}
    return hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prepare_smoke_row(row):
    return {
        "prompt": row["problem"],
        "answer": row["answer"],
        "metadata": {
            "question_key": question_key(row),
            "cohort": row["cohort"],
            "paper_id": row["paper_id"],
            "node_id": row["node_id"],
        },
    }


def validate_reward(result):
    value = result.get("reward")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 7
    ):
        raise ValueError("Rubric service returned an invalid seven-point reward")
    gap = result.get("judge_order_gap")
    if isinstance(gap, bool) or not isinstance(gap, (int, float)) or not math.isfinite(gap) or not 0 <= gap <= 7:
        raise ValueError("Rubric service returned an invalid answer-order gap")
    output = {"reward": float(value), "judge_order_gap": float(gap)}
    if "generator_truncated" in result:
        if type(result["generator_truncated"]) is not bool:
            raise ValueError("Rubric service returned an invalid generator finish classification")
        output["generator_truncated"] = float(result["generator_truncated"])
    return output


def scoring_request(sample):
    truncated = getattr(sample.status, "value", None) == "truncated"
    if final_answer(sample.response) is None and not truncated:
        raise ValueError("Unexpected unfinished thinking without a recorded model token limit")
    if not sample.metadata.get("question_key"):
        raise ValueError("TCS reward requires a pinned question identity")
    return {
        "question_key": sample.metadata["question_key"],
        "response": sample.response,
        "generator_truncated": truncated,
    }


async def reward(args, sample, **kwargs):
    # aiohttp is a runtime dependency of Miles; keep the pure data helpers usable on the VM.
    import aiohttp

    generated_path = (
        save_generated_sample(
            args.save_debug_rollout_data, sample.to_dict(), compress=os.environ.get("TCS_COMPRESS_ARTIFACTS") == "gzip"
        )
        if args.save_debug_rollout_data
        else None
    )
    try:
        if not args.rm_url:
            raise ValueError("TCS reward requires --rm-url")
        payload = scoring_request(sample)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
            async with session.post(args.rm_url, json=payload) as response:
                if response.status != 200:
                    raise RuntimeError(f"TCS rubric service HTTP {response.status}; no zero reward substituted")
                result = validate_reward(await response.json())
    except Exception as error:
        save_judge_outcome(generated_path, error_type=type(error).__name__)
        raise
    save_judge_outcome(generated_path, result=result)
    return result
