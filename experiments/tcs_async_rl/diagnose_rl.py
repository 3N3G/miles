"""Audit existing easy-only RL artifacts without sampling or changing a run.

Run with the immutable original run directory and a separate output directory.
Supply checksum-matched consumed groups under --trace-directory. Outputs are
descriptive. Repeated training visits are not a frozen-policy evaluation.
"""

import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path

from experiments.tcs_async_rl.native_metrics import compare_wandb, read_native_metrics
from experiments.tcs_async_rl.reward_hook import question_key
from experiments.tcs_async_rl.rl_artifacts import file_sha256
from experiments.tcs_async_rl.rubric_judge import parse_grade, rubric_items, rubric_output_contract


def original_scoring_payload(folder, receipt, manifest, previous):
    """Verify failure-only clarification provenance before comparing length retries."""
    payload = receipt["payload"]
    if manifest["judge_config"].get("grading_method") != "reference_guided":
        return payload
    data = json.loads(payload["messages"][1]["content"])
    if "output_contract" not in data:
        return payload
    contract = data.pop("output_contract")
    if contract != rubric_output_contract(rubric_items(data["rubric"])):
        raise ValueError("Format clarification changed the original rubric items")
    original = {**payload, "messages": [dict(message) for message in payload["messages"]]}
    original["messages"][1]["content"] = json.dumps(data, ensure_ascii=False)
    source_digest = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    path = folder / "format-retries.jsonl"
    links = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    links = [link for link in links if link["repaired_request_sha256"] == receipt["request_sha256"]]
    if not links:
        # A larger-budget retry can inherit an already verified clarification.
        if previous is None or "output_contract" not in json.loads(previous["payload"]["messages"][1]["content"]):
            raise ValueError("Judge format repair has no failed-request provenance")
        before = previous["payload"]
        if {k: v for k, v in before.items() if k != "max_completion_tokens"} != {
            k: v for k, v in payload.items() if k != "max_completion_tokens"
        }:
            raise ValueError("Judge retry changed its verified format clarification")
        return original
    names = [
        f"{receipt['identity']}-order{int(receipt['swapped'])}-{source_digest[:12]}-attempt{number}.json"
        for number in range(manifest["judge_config"]["max_attempts"])
    ]
    for link in links:
        if link["source_request_sha256"] != source_digest or link["invalid_attempt_receipts"] != names:
            raise ValueError("Judge format repair has inconsistent source attempts")
    for name in names:
        attempt = json.loads((folder / name).read_text())
        if (
            attempt["request_sha256"] != source_digest
            or attempt["payload"] != original
            or attempt["identity"] != receipt["identity"]
            or attempt["swapped"] != receipt["swapped"]
        ):
            raise ValueError("Format repair source differs from the original scoring request")
        if attempt["http_status"] == 200:
            choice = attempt["body"]["choices"][0]
            if choice["finish_reason"] != "stop":
                raise ValueError("An unfinished verdict needs a length retry, not a format repair")
            try:
                parse_grade(choice["message"]["content"], rubric_items(data["rubric"]), answer_labels=("candidate",))
            except (ValueError, TypeError, AttributeError):
                continue
            raise ValueError("A valid judge verdict was rerolled through format repair")
    return original


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def describe(values):
    return {"n": len(values), "mean": statistics.mean(values), "min": min(values), "max": max(values)}


def per_question_steps(groups):
    """Combine repeated prompt groups at the same optimizer-update coordinate."""
    grouped = collections.defaultdict(list)
    for group in groups:
        grouped[(group["optimizer_step"], group["question_key"])].append(group)
    result = []
    for members in grouped.values():
        if len(members) == 1:
            result.append(members[0])
            continue
        counts = [row.get("samples", 32) for row in members]
        total = sum(counts)
        merged = {
            **members[0],
            "samples": total,
            "merged_prompt_groups": len(members),
            "group_index": None,
            "group_indices": [row.get("group_index") for row in members],
        }
        for key in ("mean_reward", "mean_tokens", "cap_rate", "full_credit_rate", "zero_rate"):
            if all(key in row for row in members):
                merged[key] = sum(count * row[key] for count, row in zip(counts, members, strict=True)) / total
        if all("reward_std" in row for row in members):
            square_sum = sum(
                (count - 1) * row["reward_std"] ** 2 + count * (row["mean_reward"] - merged["mean_reward"]) ** 2
                for count, row in zip(counts, members, strict=True)
            )
            merged["reward_std"] = math.sqrt(square_sum / (total - 1))
            merged["conditional_answer_sampling_se"] = merged["reward_std"] / math.sqrt(total)
        result.append(merged)
    return result


def validate_group_metrics(groups, native):
    """Compare complete optimizer batches with their sample-weighted group summaries."""
    by_step = {row["optimizer_step"]: row for row in native}
    grouped = collections.defaultdict(list)
    for group in groups:
        grouped[group["optimizer_step"]].append(group)
    for step, members in grouped.items():
        total = sum(row["samples"] for row in members)
        for group_key, native_key in (
            ("mean_reward", "rollout/episode_raw_reward"),
            ("mean_tokens", "rollout/response_lengths"),
        ):
            observed = sum(row["samples"] * row[group_key] for row in members) / total
            if not math.isclose(observed, by_step[step][native_key], rel_tol=0, abs_tol=1e-8):
                raise ValueError("Audited optimizer-batch means differ from native training metrics")


def question_comparisons(groups):
    visits = collections.defaultdict(list)
    for group in per_question_steps(groups):
        visits[group["question_key"]].append(group)
    pairs = []
    for key, entries in visits.items():
        entries = sorted(entries, key=lambda x: x["optimizer_step"])
        if len(entries) < 2:
            continue
        first, last = entries[0], entries[-1]
        pairs.append(
            {
                "question_key": key,
                "paper_id": first["paper_id"],
                "first_step": first["optimizer_step"],
                "last_step": last["optimizer_step"],
                "first_reward": first["mean_reward"],
                "last_reward": last["mean_reward"],
                "difference": last["mean_reward"] - first["mean_reward"],
                "first_cap_rate": first["cap_rate"],
                "last_cap_rate": last["cap_rate"],
                "first_mean_tokens": first["mean_tokens"],
                "last_mean_tokens": last["mean_tokens"],
            }
        )
    return pairs


def publication_checks(directory, rows, native):
    events = [json.loads(line) for line in (directory / "events/main.jsonl").read_text().splitlines()]
    weights = [event for event in events if event["type"] == "inference_engine_weight_checksum"]
    changes = []
    for previous, current in zip(weights, weights[1:], strict=False):
        assert all(engine == current["engine_checksums"][0] for engine in current["engine_checksums"])
        before, after = previous["engine_checksums"][0], current["engine_checksums"][0]
        keys = [key for key in after if "/visual." not in key and "/mtp." not in key]
        changes.append(
            {
                "optimizer_step": current["rollout_id"],
                "changed_language_tensors": sum(before.get(key) != after[key] for key in keys),
                "language_tensors": len(keys),
            }
        )
    decode_lines = collections.Counter()
    for line in (directory / "rl-smoke.log").read_text().splitlines():
        if "Decode batch" in line and (match := re.search(r"pid=(\d+)", line)):
            decode_lines[match[1]] += 1
    args = json.loads((directory / "validated-native-arguments.json").read_text())["validated"]
    argument_names = [
        "lr",
        "clip_grad",
        "eps_clip",
        "eps_clip_high",
        "grpo_std_normalization",
        "rewards_normalization",
        "normalize_advantages",
        "observe_training_entropy",
        "entropy_coef",
        "use_kl_loss",
        "kl_coef",
        "mtp_detach_heads",
        "mtp_loss_scaling_factor",
        "enable_mtp_training",
        "use_sampling_support_replay",
        "rollout_top_p",
        "rollout_top_k",
        "sglang_enable_prefill_weight_versions",
    ]
    return {
        "immutable_gpu_source": json.loads((directory / "submission-plan.json").read_text())["source_commit"],
        "published_weight_states": len(weights),
        "all_three_engines_agree_at_each_publication": all(
            all(engine == event["engine_checksums"][0] for engine in event["engine_checksums"]) for event in weights
        ),
        "weight_tensor_checksum_changes": changes,
        "actor_wait_fraction_of_summed_step_time": sum(r["perf/train_wait_time"] for r in native)
        / sum(r["perf/step_time"] for r in native),
        "actor_train_seconds": sum(r["perf/actor_train_time"] for r in native),
        "actor_wait_seconds": sum(r["perf/train_wait_time"] for r in native),
        "save_seconds": sum(r.get("perf/save_model_time", 0) for r in native),
        "weight_publication_seconds": sum(r["perf/update_weights_time"] for r in native),
        "per_token_mean_version_lag": sum(
            (r["optimizer_step"] - int(v)) * n for r in rows for v, n in r["decode_tokens_by_version"].items()
        )
        / sum(r["response_tokens"] for r in rows),
        "visible_decode_log_lines_by_actor_pid": dict(decode_lines),
        "engine_utilization_limit": "Visible decode logs identify one engine actor. No GPU utilization telemetry was saved.",
        "concurrent_prompt_groups": args["async_max_concurrent_samples"] // args["n_samples_per_prompt"],
        "capped_with_positive_reward": sum(r["status"] == "truncated" and r["reward"] > 0 for r in rows),
        "capped_with_zero_reward": sum(r["status"] == "truncated" and r["reward"] == 0 for r in rows),
        "judge_gap_at_least_2": sum(r["order_gap"] >= 2 for r in rows if r["order_gap"] is not None),
        "judge_gap_at_least_3": sum(r["order_gap"] >= 3 for r in rows if r["order_gap"] is not None),
        "reference_below_seven_examples": [
            {k: r[k] for k in ["optimizer_step", "sample_index", "minimum_reference_score", "judge_identity"]}
            for r in rows
            if r["minimum_reference_score"] is not None and r["minimum_reference_score"] < 7
        ],
        "arguments": {name: args[name] for name in argument_names},
        "new_gpu_jobs": 0,
        "new_judge_calls": 0,
    }


def completed_judge_receipt(folder, identity, order):
    manifest = json.loads((folder / "service-manifest.json").read_text())
    base_tokens = manifest["judge"]["max_completion_tokens"]
    maximum_tokens = manifest["judge_config"].get("length_retry_max_completion_tokens", base_tokens)
    files = [path for path in folder.glob(f"{identity}-order{order}-*.json") if "-attempt" not in path.name]
    if not files:
        raise ValueError(f"Missing judge receipt: {identity}, order {order}")
    receipts = []
    for path in files:
        receipt = json.loads(path.read_text())
        digest = hashlib.sha256(json.dumps(receipt["payload"], sort_keys=True).encode()).hexdigest()
        if receipt["request_sha256"] != digest or path.name != f"{identity}-order{order}-{digest[:12]}.json":
            raise ValueError("Judge receipt does not match its request fingerprint")
        if receipt["http_status"] != 200 or receipt["identity"] != identity or receipt["swapped"] != bool(order):
            raise ValueError("Judge receipt has a different identity or presentation order")
        payload = receipt["payload"]
        if (
            payload["model"] != manifest["judge"]["primary_model"]
            or payload["reasoning_effort"] != manifest["judge"]["reasoning_effort"]
            or hashlib.sha256(payload["messages"][0]["content"].encode()).hexdigest() != manifest["prompt_sha256"]
        ):
            raise ValueError("Judge receipt differs from the frozen model, effort or prompt")
        receipts.append((path, receipt))
    receipts.sort(key=lambda entry: entry[1]["payload"]["max_completion_tokens"])
    originals = [
        original_scoring_payload(folder, receipt, manifest, receipts[index - 1][1] if index else None)
        for index, (_, receipt) in enumerate(receipts)
    ]
    if receipts[0][1]["payload"]["max_completion_tokens"] != base_tokens:
        raise ValueError("Judge retry is missing its original request")
    if receipts[-1][1]["payload"]["max_completion_tokens"] > maximum_tokens:
        raise ValueError("Judge retry exceeds its recorded inference budget")
    retry_file = folder / "length-retries.jsonl"
    links = [json.loads(line) for line in retry_file.read_text().splitlines()] if retry_file.exists() else []
    for index, ((path, previous), _) in enumerate(zip(receipts, receipts[1:], strict=False)):
        before, after = originals[index], originals[index + 1]
        if previous["body"]["choices"][0]["finish_reason"] != "length":
            raise ValueError("A completed valid judge verdict was regraded")
        if {k: v for k, v in before.items() if k != "max_completion_tokens"} != {
            k: v for k, v in after.items() if k != "max_completion_tokens"
        }:
            raise ValueError("Judge length retry changed the scoring request")
        if after["max_completion_tokens"] != min(2 * before["max_completion_tokens"], maximum_tokens):
            raise ValueError("Judge retry does not match the recorded token-budget expansion")
        if not any(
            Path(link["receipt"]).name == path.name
            and link["from_tokens"] == before["max_completion_tokens"]
            and link["to_tokens"] == after["max_completion_tokens"]
            for link in links
        ):
            raise ValueError("Judge length retry has no recorded provenance")
    final = receipts[-1][1]
    if final["body"]["choices"][0]["finish_reason"] != "stop":
        raise ValueError("No completed judge verdict")
    return final


def judge_receipts(directory, question, sample):
    folder = directory / "judging" / sample["metadata"]["question_key"]
    identity = sample["judge_identity"]
    score = json.loads((folder / f"{identity}-score.json").read_text())
    if score.get("grading_method") == "reference_guided":
        receipt = completed_judge_receipt(folder, identity, 0)
        request = json.loads(receipt["payload"]["messages"][1]["content"])
        request.pop("output_contract", None)  # Provenance and exact contract checked above.
        assert request == {
            "problem": question["problem"],
            "rubric": question["rubric"],
            "reference_solution": question["answer"],
            "candidate_solution": sample["final_answer"] or "",
        }
        grade = parse_grade(
            receipt["body"]["choices"][0]["message"]["content"],
            rubric_items(question["rubric"]),
            answer_labels=("candidate",),
        )["candidate"]
        assert score["scores"] == [grade["score"]]
        assert score["judge_presentations"] == sample["reward"]["judge_presentations"] == 1
        assert grade["score"] == score["reward"] == sample["reward"]["reward"]
        assert "judge_order_gap" not in score and "judge_order_gap" not in sample["reward"]
        return {"order_scores": [grade["score"]], "reference_scores": [], "candidate_decisions": [grade["items"]]}
    orders, refs, decisions = [], [], []
    for order in (0, 1):
        receipt = completed_judge_receipt(folder, identity, order)
        assert receipt["http_status"] == 200 and receipt["swapped"] == bool(order)
        choice = receipt["body"]["choices"][0]
        assert choice["finish_reason"] == "stop"
        grade = parse_grade(choice["message"]["content"], rubric_items(question["rubric"]))
        candidate_label, reference_label = ("A", "B") if order == 0 else ("B", "A")
        request = json.loads(receipt["payload"]["messages"][1]["content"])
        assert request["problem"] == question["problem"] and request["rubric"] == question["rubric"]
        assert request["answers"][reference_label] == question["answer"]
        assert request["answers"][candidate_label] == (sample["final_answer"] or "")
        orders.append(grade[candidate_label]["score"])
        refs.append(grade[reference_label]["score"])
        decisions.append(grade[candidate_label]["items"])
    assert orders == score["scores"]
    assert statistics.mean(orders) == sample["reward"]["reward"] == score["reward"]
    assert abs(orders[0] - orders[1]) == sample["reward"]["judge_order_gap"]
    return {"order_scores": orders, "reference_scores": refs, "candidate_decisions": decisions}


def trace_summary(sample):
    length = sample["response_length"]
    tokens = sample["tokens"][-length:]
    assert length > 0 and len(tokens) == length
    log_probs = sample["rollout_log_probs"]
    assert len(log_probs) == length and all(math.isfinite(x) and x <= 1e-5 for x in log_probs)
    mask = sample["loss_mask"]
    assert mask is None or (len(mask) == length and all(x in (0, 1) for x in mask))
    active = length if mask is None else sum(mask)
    support = sample["rollout_sampling_mask"]
    assert support["encoding"] == "csr"
    ids, offsets = support["ids"], support["offsets"]
    assert len(offsets) == length + 1 and offsets[0] == 0 and offsets[-1] == len(ids)
    assert all(a < b and token in ids[a:b] for token, a, b in zip(tokens, offsets, offsets[1:], strict=False))
    versions = collections.Counter()
    for call in sample["weight_versions"]:
        for span in call["spans"]:
            versions[int(span["version"])] += span["abs_end"] - span["abs_start"]
    assert sum(versions.values()) == length
    assert sample["status"] in ("completed", "truncated")
    return {
        "response_tokens": length,
        "active_loss_tokens": active,
        "mean_sampled_log_probability": statistics.mean(log_probs),
        "support_ids": len(ids),
        "tokens_with_support_size_one": sum(b - a == 1 for a, b in zip(offsets, offsets[1:], strict=False)),
        "decode_tokens_by_version": dict(versions),
        "response_characters": len(sample["response"]),
        "final_characters": len(sample["final_answer"] or ""),
    }


def read_samples(directory, traces, output, *, optimizer_steps):
    questions = {question_key(q): q for q in json.loads((directory / "input.json").read_text())["questions"]}
    archive = json.loads((directory / "archive-index.json").read_text())
    resolved = directory / "resolved-config.json"
    args = json.loads(resolved.read_text())["args"] if resolved.exists() else {}
    group_size, batch_size = args.get("n_samples_per_prompt", 32), args.get("global_batch_size", 32)
    seed = args.get("seed", 20261118)
    rows, groups, judge_cache = [], [], {}
    for step in optimizer_steps:
        name = f"rollout_data/rollout-{step}.jsonl.gz"
        path = traces / name
        assert file_sha256(path) == archive[name]["sha256"], name
        step_groups, examples = collections.defaultdict(list), {}
        selected = random.Random(seed).randrange(group_size)
        with gzip.open(path, "rt") as stream:
            for line in stream:
                sample = json.loads(line)
                assert sample["optimizer_rollout_id"] == step
                group_index = sample["group_index"]
                if batch_size > group_size and type(group_index) is not int:
                    raise ValueError("Multi-question training requires recorded prompt-group identities")
                key = sample["metadata"]["question_key"]
                question = questions[key]
                cache_key = (key, sample["judge_identity"])
                if cache_key not in judge_cache:
                    judge_cache[cache_key] = judge_receipts(directory, question, sample)
                judge = judge_cache[cache_key]
                assert sample["reward"]["reward"] == statistics.mean(judge["order_scores"])
                row = {
                    "optimizer_step": step,
                    "sample_index": sample["index"],
                    "group_index": group_index,
                    "question_key": key,
                    "paper_id": question["paper_id"],
                    "prompt_type": question["prompt_type"],
                    "difficulty": question.get("difficulty"),
                    "reward": sample["reward"]["reward"],
                    "order0": judge["order_scores"][0],
                    "order1": judge["order_scores"][1] if len(judge["order_scores"]) == 2 else None,
                    "order_gap": (
                        abs(judge["order_scores"][0] - judge["order_scores"][1])
                        if len(judge["order_scores"]) == 2
                        else None
                    ),
                    "minimum_reference_score": min(judge["reference_scores"], default=None),
                    "status": sample["status"],
                    "judge_identity": sample["judge_identity"],
                    **trace_summary(sample),
                }
                rows.append(row)
                group = step_groups[group_index]
                group.append(row)
                if len(group) - 1 == selected:
                    example = {
                        "optimizer_step": step,
                        "sample_index": sample["index"],
                        "selection": f"Random({seed}).randrange({group_size}) within each group",
                        "question": question,
                        "response": sample["response"],
                        "final_answer": sample["final_answer"],
                        "reward": sample["reward"],
                        "judge": judge,
                        "trace": row,
                    }
                    suffix = f"-group-{group_index}" if batch_size > group_size else ""
                    target = output / f"example-step-{step}{suffix}.json"
                    save(target, example)
                    examples[group_index] = target.name
        assert sum(len(group) for group in step_groups.values()) == batch_size
        for group_index, group in step_groups.items():
            assert len(group) == group_size and len({s["question_key"] for s in group}) == 1
            assert len({s["sample_index"] for s in group}) == group_size
            scores = [s["reward"] for s in group]
            groups.append(
                {
                    "optimizer_step": step,
                    "group_index": group_index,
                    "question_key": group[0]["question_key"],
                    "paper_id": group[0]["paper_id"],
                    "prompt_type": group[0]["prompt_type"],
                    "difficulty": group[0]["difficulty"],
                    "samples": len(group),
                    "mean_reward": statistics.mean(scores),
                    "reward_std": statistics.stdev(scores),
                    "conditional_answer_sampling_se": statistics.stdev(scores) / math.sqrt(len(group)),
                    "zero_rate": scores.count(0) / len(group),
                    "full_credit_rate": scores.count(7) / len(group),
                    "cap_rate": sum(s["status"] == "truncated" for s in group) / len(group),
                    "mean_tokens": statistics.mean(s["response_tokens"] for s in group),
                    "mean_order_gap": (
                        statistics.mean(s["order_gap"] for s in group if s["order_gap"] is not None)
                        if any(s["order_gap"] is not None for s in group)
                        else None
                    ),
                    "order_disagreement_rate": (
                        sum(s["order_gap"] > 0 for s in group if s["order_gap"] is not None) / len(group)
                        if all(s["order_gap"] is not None for s in group)
                        else None
                    ),
                    "example": examples[group_index],
                }
            )
        print(
            f"Validated group {step}: samples, judge receipts, log-probs, loss masks, support and versions",
            flush=True,
        )
    return rows, groups


def analyze(directory, traces, output):
    output.mkdir(parents=True, exist_ok=True)
    native = read_native_metrics(directory)
    save(output / "native-metrics.json", native)
    rows, groups = read_samples(directory, traces, output, optimizer_steps=[row["optimizer_step"] for row in native])
    if len(groups) != len(native):
        raise ValueError("Use audit_live and summarize_continuation for multiple prompt groups per update")
    save(output / "samples.json", rows)
    save(output / "groups.json", groups)
    save(output / "source-checks.json", publication_checks(directory, rows, native))
    pairs = question_comparisons(groups)
    save(output / "same-question-comparisons.json", pairs)
    with (output / "native-metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in native for key in row}))
        writer.writeheader()
        writer.writerows(native)
    history = json.loads((output / "wandb-full-history.json").read_text())["rows"]
    wandb_audit = [
        compare_wandb(native, history, metric=metric, step_key=axis)
        for metric, axis in [
            ("train/pg_loss", "train/step"),
            ("rollout/raw_reward", "rollout/step"),
            ("rollout/episode_raw_reward", "rollout/step"),
        ]
    ]
    save(output / "wandb-comparison.json", wandb_audit)
    for group, metrics in zip(groups, native, strict=True):
        assert group["mean_reward"] == metrics["rollout/raw_reward"] == metrics["rollout/episode_raw_reward"]
        assert group["mean_tokens"] == metrics["rollout/response_lengths"]
    metric_summary = {key: describe([r[key] for r in native if key in r]) for key in native[0]}
    # Some metrics, such as checkpoint save time, appear only on specific steps.
    metric_summary.update(
        {
            key: describe([r[key] for r in native if key in r])
            for key in {k for r in native for k in r} - set(metric_summary)
        }
    )
    tokens = sum(r["response_tokens"] for r in rows)
    caps = [r for r in rows if r["status"] == "truncated"]
    first = {g["question_key"]: g for g in reversed(groups)}
    matched_first = statistics.mean(p["first_reward"] for p in pairs)
    matched_last = statistics.mean(p["last_reward"] for p in pairs)
    summary = {
        "updates": len(native),
        "answers": len(rows),
        "questions": len(first),
        "question_visit_counts": dict(
            collections.Counter(collections.Counter(g["question_key"] for g in groups).values())
        ),
        "first_pass_mean_reward": statistics.mean(g["mean_reward"] for g in groups[:20]),
        "later_visits_mean_reward": statistics.mean(g["mean_reward"] for g in groups[20:]),
        "matched_first_mean_reward": matched_first,
        "matched_last_mean_reward": matched_last,
        "same_question_mean_change": matched_last - matched_first,
        "questions_increased": sum(p["difference"] > 0 for p in pairs),
        "questions_decreased": sum(p["difference"] < 0 for p in pairs),
        "reward_group_range": [min(g["mean_reward"] for g in groups), max(g["mean_reward"] for g in groups)],
        "varied_groups": sum(g["reward_std"] > 0 for g in groups),
        "answer_sampling_se_range": [
            min(g["conditional_answer_sampling_se"] for g in groups),
            max(g["conditional_answer_sampling_se"] for g in groups),
        ],
        "zero_answers": sum(r["reward"] == 0 for r in rows),
        "full_credit_answers": sum(r["reward"] == 7 for r in rows),
        "judge_order_presentations_measured": sum(r["order_gap"] is not None for r in rows),
        "judge_order_disagreements": sum(r["order_gap"] > 0 for r in rows if r["order_gap"] is not None),
        "mean_judge_order_gap": (
            statistics.mean(r["order_gap"] for r in rows if r["order_gap"] is not None)
            if any(r["order_gap"] is not None for r in rows)
            else None
        ),
        "maximum_judge_order_gap": max((r["order_gap"] for r in rows if r["order_gap"] is not None), default=None),
        "reference_below_full_credit": sum(
            r["minimum_reference_score"] < 7 for r in rows if r["minimum_reference_score"] is not None
        ),
        "generated_tokens": tokens,
        "mean_tokens_per_answer": tokens / len(rows),
        "capped_answers": len(caps),
        "capped_token_fraction": sum(r["response_tokens"] for r in caps) / tokens,
        "final_response_character_fraction": sum(r["final_characters"] for r in rows)
        / sum(r["response_characters"] for r in rows),
        "tokens_with_support_size_one_fraction": sum(r["tokens_with_support_size_one"] for r in rows) / tokens,
        "active_loss_tokens": sum(r["active_loss_tokens"] for r in rows),
        "native_metrics": metric_summary,
        "limits": [
            "One training seed and only seven repeated questions. No frozen before/after evaluation.",
            "Judge answer-order gap is not a human agreement estimate. Human calibration remains absent.",
            "Logged entropy is disabled, not measured zero entropy. Reference-policy KL was not measured.",
            "Per-token KL/ESS cannot certify full-trajectory importance weights at these lengths.",
            "No per-parameter update norms or backbone/MTP gradient decomposition were logged.",
        ],
    }
    save(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--trace-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    summary = analyze(cli.run_directory, cli.trace_directory, cli.output)
    print(json.dumps({k: v for k, v in summary.items() if k != "native_metrics"}, indent=2))


if __name__ == "__main__":
    main()
