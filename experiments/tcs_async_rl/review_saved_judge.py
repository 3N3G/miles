"""Review pre-adjudicated saved proofs through an existing VM judge service.

The service owns credentials, concurrency and the existing shared spending cap.
This client validates its prompt fingerprint and preserves prior review outputs.
It changes no GPU job, training setting, prompt, rubric or assistant label.
"""

import argparse
import asyncio
import datetime
import hashlib
import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from experiments.tcs_async_rl.diagnose_rl import save
from experiments.tcs_async_rl.qualify_judge_repair import review_labels
from experiments.tcs_async_rl.rubric_judge import rubric_items


class ExistingService:
    def __init__(self, directory, config_path):
        approved = json.loads((directory / "approved-config.json").read_text())
        if config_path.name != approved["judge_config"]:
            raise ValueError("Judge config differs from the approved run")
        self.url = approved["overrides"]["rm_url"]
        parsed = urllib.parse.urlsplit(self.url)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.path != "/score":
            raise ValueError("Use the existing VM loopback scorer")
        config = json.loads(config_path.read_text())
        prompt = config_path.parent / config["prompt_file"]
        prompt_sha = hashlib.sha256(prompt.read_bytes()).hexdigest()
        expected = json.loads((config_path.parent / config["training_config"]).read_text())["judge"]
        root = directory / "judging"
        collection = json.loads((root / "collection-manifest.json").read_text())
        self.scorers = {}
        for key in collection["questions"]:
            folder = root / key
            manifest = json.loads((folder / "service-manifest.json").read_text())
            if (
                manifest["judge_config"] != config
                or manifest["judge"] != expected
                or manifest["prompt_sha256"] != prompt_sha
            ):
                raise ValueError("Existing service receipts use a different grader")
            self.scorers[key] = SimpleNamespace(directory=folder)
        health_url = urllib.parse.urlunsplit(parsed._replace(path="/health"))
        with urllib.request.urlopen(health_url, timeout=30) as response:
            self.health = json.load(response)
        if (
            not self.health["ready"]
            or self.health["question_key"] != collection["question_set_key"]
            or self.health["model"] != expected["primary_model"]
            or self.health["reasoning_effort"] != expected["reasoning_effort"]
        ):
            raise ValueError("Live service differs from the expected question collection or model")
        self.prompt_sha256 = prompt_sha

    def _post(self, payload):
        request = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=1800) as response:
            return json.load(response)

    async def score(self, payload):
        return await asyncio.to_thread(self._post, payload)


def validate_labels(path, expected_sha):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Assistant labels changed after their pre-call freeze")
    labels = json.loads(raw)
    for case in labels["cases"]:
        source = Path(case["source_file"]).read_bytes()
        if hashlib.sha256(source).hexdigest() != case["source_sha256"]:
            raise ValueError("Original saved proof changed")
        example = json.loads(source)
        if "original_example" in example:
            example = example["original_example"]
        items = rubric_items(example["question"]["rubric"])
        decisions = case["candidate_items"]
        if len(decisions) != len(items) or {d["item"] for d in decisions} != {i for i, _ in items}:
            raise ValueError("Assistant labels do not cover the supplied rubric")
        if any(type(d["satisfied"]) is not bool for d in decisions):
            raise ValueError("Assistant item labels must be Boolean")
    return labels


async def review(cli):
    output = cli.output_directory
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("assistant-label-validation.json", "audit-runtime-provenance.json")):
        raise FileExistsError("Preserve prior reviews and choose a new output directory")
    labels = validate_labels(cli.labels, cli.labels_sha256)
    service = ExistingService(cli.run_directory, cli.judge_config)
    ledger = cli.run_directory / "judging/budget.jsonl"
    before = ledger.read_bytes()
    provenance = {
        "at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "analysis_source_commit": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"], text=True
        ).strip(),
        "labels_sha256": cli.labels_sha256,
        "prompt_sha256": service.prompt_sha256,
        "service_health": service.health,
        "shared_ledger_sha256_before": hashlib.sha256(before).hexdigest(),
        "new_solver_generations": 0,
        "new_gpu_jobs": 0,
    }
    save(output / "audit-runtime-provenance.json", provenance)
    reviewed = await review_labels(service, labels, output)
    after = ledger.read_bytes()
    if not after.startswith(before):
        raise ValueError("Shared judge ledger was replaced during review")
    new_entries = [json.loads(line) for line in after[len(before) :].splitlines()]
    prefixes = tuple(case["result"]["identity"] + "-" for case in reviewed)
    summary = {
        "cases": len(reviewed),
        "item_decisions": sum(2 * len(case["candidate_items"]) for case in reviewed),
        "item_mismatches": sum(len(case["differences"]) for case in reviewed),
        "candidate_scores": {case["case"]: case["result"]["scores"] for case in reviewed},
        "reference_scores_below_seven": sum(
            order["reference"]["score"] < 7 for case in reviewed for order in case["verdicts"]
        ),
        "new_reserved_audit_calls": sum(
            row["event"] == "reserved" and row["call_id"].startswith(prefixes) for row in new_entries
        ),
        "shared_ledger_usd_change_during_review": sum(row["usd_delta"] for row in new_entries),
        "shared_usd_spent_or_reserved": sum(json.loads(line)["usd_delta"] for line in after.splitlines()),
        "limits": "Assistant diagnostic labels on existing training questions. Repeated item decisions are not independent examples or human accuracy.",
    }
    save(output / "additional-audit-summary.json", summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-directory", "judge-config", "labels", "output-directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--labels-sha256", required=True)
    asyncio.run(review(parser.parse_args()))


if __name__ == "__main__":
    main()
