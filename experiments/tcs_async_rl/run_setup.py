"""Freeze and publish the exact judge contract without adding metric history."""

import hashlib
import json

from experiments.tcs_async_rl.rl_artifacts import _atomic_bytes
from experiments.tcs_async_rl.rubric_judge import messages

PUBLIC_JUDGE_FIELDS = (
    "grading_method",
    "answer_extraction",
    "max_concurrency",
    "max_attempts",
    "timeout_seconds",
    "length_retry_max_completion_tokens",
    "usd_cap",
)


def freeze_judge_setup(directory, config_path, config):
    """Copy the used prompt, not a later repository version with the same name."""
    prompt = (config_path.parent / config["prompt_file"]).read_bytes()
    _atomic_bytes(directory / "judge-system-prompt.txt", prompt)
    row = {"problem": "{{problem}}", "rubric": "{{original_rubric}}", "answer": "{{reference_solution}}"}
    template = messages(
        row,
        "{{candidate_final_answer}}",
        swapped=False,
        system_prompt=prompt.decode(),
        grading_method=config.get("grading_method", "anonymous_pair"),
    )
    _atomic_bytes(directory / "judge-request-template.json", (json.dumps(template, indent=2) + "\n").encode())
    return {key: config[key] for key in PUBLIC_JUDGE_FIELDS if key in config}


def provenance_config(directory, resolved, approved):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    proposal = manifest.get("training_proposal", {})
    result = {
        "code_commit": manifest.get("commit"),
        "model": {key: proposal.get("model", {}).get(key) for key in ("repo_id", "revision")},
        "data": {
            key: proposal.get("data", {}).get(key)
            for key in ("repo_id", "revision", "eligible_configs", "held_out_configs")
        },
        "resume": {key: approved.get("resume", {}).get(key) for key in ("source_run", "iteration")},
        "allocation": {
            key: approved.get("allocation", {}).get(key)
            for key in ("partition", "nodes", "gpus", "cpus_per_task", "walltime")
        },
        "container_uri": manifest.get("qualification", {}).get("container", {}).get("uri"),
        "runtime_versions": resolved.get("runtime_versions", {}),
        "judge_runtime": {key: resolved.get("judge_runtime", {}).get(key) for key in PUBLIC_JUDGE_FIELDS},
        "reward_definition": "Sum original item weights for satisfied Boolean items; raw reward in [0,7].",
        "metric_axis": "audit/optimizer_step; _step counts publisher rows, not optimizer updates.",
        "artifact_policy": "Full generation, final-answer extraction and judge receipts in private HF; no checkpoints uploaded.",
    }
    prompt_path = directory / "judge-system-prompt.txt"
    if prompt_path.exists():
        prompt = prompt_path.read_bytes()
        if hashlib.sha256(prompt).hexdigest() != resolved.get("judge_prompt_sha256"):
            raise ValueError("Frozen judge prompt differs from the actual resolved prompt hash")
        result["judge_system_prompt"] = prompt.decode()
    return result


def publish_run_setup(run, directory, config):
    """Print once and attach files through the existing single writer, without run.log."""
    if "judge_system_prompt" not in config:
        return
    receipt = directory / "setup-publication.json"
    identity = {key: config[key] for key in ("resolved_config_sha256", "judge_prompt_sha256")}
    if receipt.exists():
        if json.loads(receipt.read_text()) != identity:
            raise ValueError("Published setup changed within one training run")
        return
    folder = directory / "setup"
    folder.mkdir(exist_ok=True)
    card = (
        "# Training run setup\n\n"
        "Actual optimizer accumulation is audit/train/num_microbatches at each update.\n\n"
        "## Configuration and provenance\n\n```json\n"
        + json.dumps({key: value for key, value in config.items() if key != "judge_system_prompt"}, indent=2)
        + "\n```\n\n## Exact judge system prompt\n\n```text\n"
        + config["judge_system_prompt"]
        + "\n```\n"
    )
    files = {"RUN_SETUP.md": card, "training-hyperparameters.json": json.dumps(config, indent=2) + "\n"}
    for name, value in files.items():
        path = folder / name
        _atomic_bytes(path, value.encode())
        run.save(str(path), base_path=str(directory), policy="now")
    for name in ("judge-system-prompt.txt", "judge-request-template.json"):
        path = directory / name
        if path.exists():
            run.save(str(path), base_path=str(directory), policy="now")
    print(card, flush=True)
    _atomic_bytes(receipt, (json.dumps(identity, indent=2) + "\n").encode())
