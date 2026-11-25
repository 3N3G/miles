"""Grade saved final answers on the VM, with explicit receipts and a spending cap."""

import argparse
import asyncio
import datetime
import fcntl
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

from experiments.tcs_async_rl.group_inference import _append_json
from experiments.tcs_async_rl.judge_client import load_credentials, request_chat
from experiments.tcs_async_rl.rubric_judge import messages, parse_grade, rubric_items

CONFIG = Path(__file__).with_name("judge_group_v1.json")


def usage_cost(usage, config):
    prompt = usage["prompt_tokens"]
    completion = usage["completion_tokens"]
    long = prompt > config["long_input_threshold_tokens"]
    return (
        prompt * config["input_usd_per_million"] * (config["long_input_multiplier"] if long else 1)
        + completion * config["output_usd_per_million"] * (config["long_output_multiplier"] if long else 1)
    ) / 1e6


def reserve_cost(payload, config):
    # UTF-8 byte count bounds input token count conservatively. Include the JSON
    # envelope, a fixed framing allowance, and worst applicable long-input rates.
    bound = len(json.dumps(payload, ensure_ascii=False).encode()) + 4096
    return usage_cost({"prompt_tokens": bound, "completion_tokens": payload["max_completion_tokens"]}, config)


class Budget:
    def __init__(self, directory, config):
        self.directory, self.config = directory, config
        self.path = directory / "budget.jsonl"
        self.total = 0.0
        if self.path.exists():
            self.total = sum(json.loads(line)["usd_delta"] for line in self.path.read_text().splitlines())

    def reserve(self, amount, call_id):
        self._record(amount, "reserved", call_id, enforce_cap=True)

    def settle(self, reserved, actual, call_id):
        self._record(actual - reserved, "settled", call_id)

    def _record(self, delta, event, call_id, enforce_cap=False):
        with (self.directory / "budget.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.total = (
                sum(json.loads(line)["usd_delta"] for line in self.path.read_text().splitlines())
                if self.path.exists()
                else 0.0
            )
            if enforce_cap and self.total + delta > self.config["usd_cap"]:
                raise RuntimeError("Judge spending cap reached; no request sent")
            self.total += delta
            _append_json(
                self.path,
                {
                    "at": datetime.datetime.now(datetime.UTC).isoformat(),
                    "event": event,
                    "call_id": call_id,
                    "usd_delta": delta,
                    "usd_total": self.total,
                },
            )


async def grade_one(
    *, identity, candidate, row, swapped, judge, config, prompt, key, budget, semaphore, directory, clarify_items=False
):
    method = config.get("grading_method", "anonymous_pair")
    payload = {
        "model": judge["primary_model"],
        "reasoning_effort": judge["reasoning_effort"],
        "max_completion_tokens": judge["max_completion_tokens"],
        "response_format": {"type": "json_object"},
        "messages": messages(
            row, candidate, swapped=swapped, system_prompt=prompt, grading_method=method, clarify_items=clarify_items
        ),
    }
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    output = directory / f"{identity}-order{int(swapped)}-{fingerprint[:12]}.json"
    needs_format_repair = False
    if output.exists():
        receipt = json.loads(output.read_text())
    else:
        async with semaphore:
            for attempt in range(config["max_attempts"]):
                call_id = f"{identity}-{int(swapped)}-{fingerprint[:12]}-{attempt}"
                attempt_name = f"{identity}-order{int(swapped)}-{fingerprint[:12]}-attempt{attempt}.json"
                attempt_file = directory / attempt_name
                if attempt_file.exists():
                    receipt = json.loads(attempt_file.read_text())
                    if receipt.get("request_sha256") != fingerprint or receipt.get("payload") != payload:
                        raise ValueError("Cached judge attempt differs from the requested payload")
                else:
                    reservation = reserve_cost(payload, config)
                    budget.reserve(reservation, call_id)
                    receipt = await asyncio.to_thread(
                        request_chat, payload, key=key, endpoint=config["endpoint"], timeout=config["timeout_seconds"]
                    )
                    receipt.update(
                        identity=identity,
                        swapped=swapped,
                        request_sha256=fingerprint,
                        attempt=attempt,
                        payload=payload,
                    )
                    usage = receipt.get("body", {}).get("usage")
                    if usage is not None:
                        budget.settle(reservation, usage_cost(usage, config), call_id)
                    attempt_file.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
                if receipt.get("http_status") == 200:
                    choice = receipt["body"]["choices"][0]
                    if choice.get("finish_reason") == "stop":
                        labels = ("candidate",) if method == "reference_guided" else ("A", "B")
                        try:
                            parse_grade(
                                choice["message"]["content"], rubric_items(row["rubric"]), answer_labels=labels
                            )
                        except (ValueError, TypeError, AttributeError):
                            # Retry only an unusable schema, never a valid grade.
                            # The paid raw attempt and unchanged request remain auditable.
                            retry_file = directory / "malformed-verdict-retries.jsonl"
                            event = {"receipt": attempt_name, "request_sha256": fingerprint, "attempt": attempt}
                            previous = (
                                [json.loads(line) for line in retry_file.read_text().splitlines()]
                                if retry_file.exists()
                                else []
                            )
                            if event not in previous:
                                _append_json(retry_file, event)
                            if attempt + 1 == config["max_attempts"]:
                                if method == "reference_guided" and not clarify_items:
                                    needs_format_repair = True
                                    break
                                raise RuntimeError(
                                    "Malformed judge verdict retries exhausted; no zero substituted"
                                ) from None
                            await asyncio.sleep(min(2**attempt, 30))
                            continue
                    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
                    break
                status = receipt.get("http_status")
                if status is not None and status not in (408, 429) and status < 500:
                    raise RuntimeError(f"Judge HTTP {status}; no zero score substituted")
                if attempt + 1 == config["max_attempts"]:
                    raise RuntimeError("Judge retries exhausted; no zero score substituted")
                await asyncio.sleep(min(2**attempt, 30))
    if needs_format_repair:
        # Release the concurrency permit before the bounded failure-only retry.
        # Ordinary requests and every valid cached grade retain their exact bytes.
        repaired_payload = {
            **payload,
            "messages": messages(
                row, candidate, swapped=swapped, system_prompt=prompt, grading_method=method, clarify_items=True
            ),
        }
        retry_file = directory / "format-retries.jsonl"
        event = {
            "source_request_sha256": fingerprint,
            "repaired_request_sha256": hashlib.sha256(
                json.dumps(repaired_payload, sort_keys=True).encode()
            ).hexdigest(),
            "invalid_attempt_receipts": [
                f"{identity}-order{int(swapped)}-{fingerprint[:12]}-attempt{number}.json"
                for number in range(config["max_attempts"])
            ],
        }
        previous = [json.loads(line) for line in retry_file.read_text().splitlines()] if retry_file.exists() else []
        if event not in previous:
            _append_json(retry_file, event)
        return await grade_one(
            identity=identity,
            candidate=candidate,
            row=row,
            swapped=swapped,
            judge=judge,
            config=config,
            prompt=prompt,
            key=key,
            budget=budget,
            semaphore=semaphore,
            directory=directory,
            clarify_items=True,
        )
    choice = receipt["body"]["choices"][0]
    retry_limit = config.get("length_retry_max_completion_tokens", judge["max_completion_tokens"])
    if choice.get("finish_reason") == "length" and judge["max_completion_tokens"] < retry_limit:
        # Only an unfinished verdict gets more inference tokens. Keep every
        # receipt, use the same rubric/model/effort, and never retry a valid score.
        expanded = min(2 * judge["max_completion_tokens"], retry_limit)
        _append_json(
            directory / "length-retries.jsonl",
            {"receipt": str(output), "from_tokens": judge["max_completion_tokens"], "to_tokens": expanded},
        )
        return await grade_one(
            identity=identity,
            candidate=candidate,
            row=row,
            swapped=swapped,
            judge={**judge, "max_completion_tokens": expanded},
            config=config,
            prompt=prompt,
            key=key,
            budget=budget,
            semaphore=semaphore,
            directory=directory,
            clarify_items=clarify_items,
        )
    if choice.get("finish_reason") != "stop":
        raise ValueError("Judge did not finish its verdict; not a zero score")
    labels = ("candidate",) if method == "reference_guided" else ("A", "B")
    grade = parse_grade(choice["message"]["content"], rubric_items(row["rubric"]), answer_labels=labels)
    return {
        "identity": identity,
        "swapped": swapped,
        "candidate": grade["candidate"] if method == "reference_guided" else grade["B" if swapped else "A"],
        "reference": None if method == "reference_guided" else grade["A" if swapped else "B"],
        "receipt": str(output),
    }


async def grade_group(*, run_dir, input_file, config_path, key_file, controls_only):
    config = json.loads(config_path.read_text())
    training = json.loads((config_path.parent / config["training_config"]).read_text())
    judge = training["judge"]
    row = json.loads(input_file.read_text())
    prompt = (config_path.parent / config["prompt_file"]).read_text()
    directory = run_dir / "judging"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "code_commit": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True
        ).strip(),
        "judge_config": config,
        "judge": judge,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }
    (directory / "grader-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    key = load_credentials(key_file)[judge["api_key_env"]]
    budget = Budget(directory, config)
    semaphore = asyncio.Semaphore(config["max_concurrency"])
    cases = [("control-reference", row["answer"]), ("control-empty", ""), ("control-irrelevant", "The answer is 42.")]
    responses = []
    if not controls_only:
        responses = [json.loads(line) for line in (run_dir / "responses.jsonl").read_text().splitlines()]
        cases += [(f"sample-{r['sample_index']:02d}", r["final_answer"]) for r in responses if r["final_answer"]]
    results = await asyncio.gather(
        *(
            grade_one(
                identity=identity,
                candidate=candidate,
                row=row,
                swapped=swapped,
                judge=judge,
                config=config,
                prompt=prompt,
                key=key,
                budget=budget,
                semaphore=semaphore,
                directory=directory,
            )
            for identity, candidate in cases
            for swapped in (False, True)
        )
    )
    means = {
        identity: statistics.mean(r["candidate"]["score"] for r in results if r["identity"] == identity)
        for identity, _ in cases
    }
    scores = [score for name, score in means.items() if name.startswith("sample-")]
    summary = {
        "status": config["grade_status"],
        "model": judge["primary_model"],
        "reasoning_effort": judge["reasoning_effort"],
        "scores_by_case": means,
        "scored_responses": len(scores),
        "incomplete_or_empty_final_answers": sum(not r["final_answer"] for r in responses),
        "all_zero": all(score == 0 for score in scores) if scores else None,
        "uniform_scores": len(set(scores)) == 1 if scores else None,
        "population_score_variance": statistics.pvariance(scores) if scores else None,
        "usd_spent_or_reserved": budget.total,
        "independent_judge_status": "not_run_by_this_command",
        "human_calibration": "pending",
        "grades": results,
    }
    sampling_file = run_dir / "sampling-parameters.json"
    if sampling_file.exists():
        summary["generation_sampling"] = json.loads(sampling_file.read_text())
    target = run_dir / ("control-grades.json" if controls_only else "group-grades.json")
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "grades"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--key-file", type=Path, default=Path.home() / ".config/api-keys.env")
    parser.add_argument("--controls-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(
        grade_group(
            run_dir=args.run_dir,
            input_file=args.input_file,
            config_path=args.config,
            key_file=args.key_file,
            controls_only=args.controls_only,
        )
    )


if __name__ == "__main__":
    main()
