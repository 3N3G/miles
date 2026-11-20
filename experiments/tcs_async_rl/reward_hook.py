"""Per-sample Miles hook for the VM-owned TCS rubric service."""

import hashlib
import json
import math

from experiments.tcs_async_rl.group_inference import final_answer


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
    return {"reward": float(value), "judge_order_gap": float(gap)}


async def reward(args, sample, **kwargs):
    # aiohttp is a runtime dependency of Miles; keep the pure data helpers usable on the VM.
    import aiohttp

    if final_answer(sample.response) is None:
        raise ValueError("Unfinished thinking has no grade; no zero reward substituted")
    if not args.rm_url or not sample.metadata.get("question_key"):
        raise ValueError("TCS reward requires --rm-url and a pinned question identity")
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as session:
        async with session.post(
            args.rm_url, json={"question_key": sample.metadata["question_key"], "response": sample.response}
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"TCS rubric service HTTP {response.status}; no zero reward substituted")
            return validate_reward(await response.json())
