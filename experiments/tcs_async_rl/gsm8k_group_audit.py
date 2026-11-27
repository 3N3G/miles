"""Audit question-wise reward normalization on an actual portable training batch."""

import argparse
import ast
import collections
import dataclasses
import gzip
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Any

from experiments.tcs_async_rl.gsm8k_artifacts import write_json

SOURCE_FILE = "miles/ray/rollout/train_data_conversion.py"
ROOT = Path(__file__).resolve().parents[2]


@dataclasses.dataclass(frozen=True)
class RewardSample:
    index: int
    group_index: int
    reward: float
    tokens: list[int]
    rollout_id: None = None

    def get_reward_value(self, args):
        return self.reward


def source_at(commit):
    return subprocess.check_output(["git", "show", f"{commit}:{SOURCE_FILE}"], cwd=ROOT, text=True)


def production_helpers(source):
    # Load the production pure functions without importing the GPU runtime tree.
    import torch

    names = {"_reward_group_segments", "_normalize_rewards_by_rollout", "_post_process_rewards", "_package_shards"}
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise ValueError("The frozen normalization contract changed")
    namespace = {"torch": torch, "Any": Any, "Sample": RewardSample}
    exec(compile(ast.Module(body=functions, type_ignores=[]), SOURCE_FILE, "exec"), namespace)
    return namespace


def independent_advantages(samples):
    grouped = collections.defaultdict(list)
    for sample in samples:
        grouped[sample.group_index].append(sample)
    expected = {}
    for group in grouped.values():
        mean = sum(sample.reward for sample in group) / len(group)
        std = math.sqrt(sum((sample.reward - mean) ** 2 for sample in group) / (len(group) - 1))
        for sample in group:
            expected[sample.index] = (sample.reward - mean) / (std + 1e-6) if std else sample.reward - mean
    return expected, grouped


def audit(directory, batch, *, compare_source=None):
    import torch

    torch.set_num_threads(1)
    commit = json.loads((directory / "manifest.json").read_text())["commit"]
    source = source_at(commit)
    if compare_source is not None and source != source_at(compare_source):
        raise ValueError("Compared runs use different normalization code")
    functions = production_helpers(source)
    parsed = json.loads((directory / "validated-native-arguments.json").read_text())["validated"]
    args = argparse.Namespace(**parsed)
    if not (args.advantage_estimator == "grpo" and args.rewards_normalization and args.grpo_std_normalization):
        raise ValueError("This independent arithmetic check expects the approved GRPO normalization")
    with gzip.open(batch, "rt") as stream:
        rows = list(map(json.loads, stream))
    samples = [RewardSample(**{key: row[key] for key in ("index", "group_index", "reward", "tokens")}) for row in rows]
    if len(samples) != args.global_batch_size or len({s.index for s in samples}) != len(samples):
        raise ValueError("The actual batch is incomplete or contains repeated sample indices")
    expected, groups = independent_advantages(samples)
    errors = []
    for permuted in (False, True):
        ordered = samples.copy()
        if permuted:
            random.Random(20261126).shuffle(ordered)
        raw, normalized = functions["_post_process_rewards"](args, ordered, None)
        errors.extend(abs(value - expected[s.index]) for s, value in zip(ordered, normalized, strict=True))
        data = {"tokens": [s.tokens for s in ordered], "rewards": normalized, "sample_indices": [s.index for s in ordered], "raw_reward": raw}
        for dp_size in (1, 2):
            partitions = [list(range(rank, len(ordered), dp_size)) for rank in range(dp_size)]
            shards = functions["_package_shards"](args, data, partitions)
            recovered = {index: value for shard in shards for index, value in zip(shard["sample_indices"], shard["rewards"], strict=True)}
            if recovered.keys() != expected.keys():
                raise ValueError("Sharding changed the sample set")
            errors.extend(abs(recovered[key] - expected[key]) for key in expected)
    if max(errors) >= 1e-5:
        raise ValueError("Question normalization differs from the independent arithmetic")
    result = {
        "kind": "CPU audit of production transformations on actual binary rewards; no new model run",
        "source_file": SOURCE_FILE,
        "source_commit": commit,
        "compared_source_commit": compare_source,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "batch_file": batch.name,
        "batch_sha256": hashlib.sha256(batch.read_bytes()).hexdigest(),
        "samples": len(samples),
        "groups": len(groups),
        "group_sizes": sorted({len(g) for g in groups.values()}),
        "maximum_absolute_error": max(errors),
        "permutation_seed": 20261126,
        "data_shard_counts_checked": [1, 2],
        "parser_defaults": {key: parsed[key] for key in ("rewards_normalization", "grpo_std_normalization", "balance_data", "kl_coef", "kl_loss_coef")},
        "limitation": "Assumes the approved single-turn trajectory format. Tests frozen CPU transformations, not a new GPU optimizer update.",
        "gpu_jobs_submitted": 0,
    }
    write_json(directory / "reward-group-audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--compare-source")
    args = parser.parse_args()
    print(json.dumps(audit(args.directory, args.batch, compare_source=args.compare_source), indent=2))
