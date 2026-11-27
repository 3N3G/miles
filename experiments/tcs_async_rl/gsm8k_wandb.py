"""Publish benchmark evaluations through the existing serialized native writer."""

import json

from experiments.tcs_async_rl.gsm8k_artifacts import write_json


def evaluation_files(directory):
    return sorted((directory / "evaluations").glob("*.json"))


def publish_evaluations(run, directory):
    receipt = directory / "wandb-evaluation-delivery.json"
    previous = json.loads(receipt.read_text()) if receipt.exists() else []
    delivered = list(previous)
    run.define_metric("control/eval_optimizer_updates")
    run.define_metric("control/eval/*", step_metric="control/eval_optimizer_updates")
    for path in evaluation_files(directory):
        if path.name in delivered:
            continue
        point = json.loads(path.read_text())
        run.log({
            "control/eval_optimizer_updates": point["optimizer_updates"],
            **{f"control/eval/{key}": point[key] for key in
               ("accuracy", "valid_box_fraction", "capped_fraction", "mean_response_tokens")},
        })
        delivered.append(path.name)
    return delivered


def acknowledge_evaluations(directory, delivered):
    write_json(directory / "wandb-evaluation-delivery.json", delivered)


def publish_setup(run, directory, config):
    receipt = directory / "setup-publication.json"
    identity = {"resolved_config_sha256": config["resolved_config_sha256"]}
    if receipt.exists():
        if json.loads(receipt.read_text()) != identity:
            raise ValueError("Benchmark setup changed within one run")
        return
    folder = directory / "setup"
    folder.mkdir(exist_ok=True)
    path = folder / "training-hyperparameters.json"
    write_json(path, config)
    run.save(str(path), base_path=str(directory), policy="now")
    prompt = directory / "formatted-prompt.txt"
    run.save(str(prompt), base_path=str(directory), policy="now")
    print("GSM8K setup. Numeric reward only. No API judge.\n" + json.dumps(config, indent=2), flush=True)
    write_json(receipt, identity)


def status_summary(directory):
    points = [json.loads(p.read_text()) for p in evaluation_files(directory)]
    if not points:
        return {}
    first = min(points, key=lambda p: p["optimizer_updates"])
    last = max(points, key=lambda p: p["optimizer_updates"])
    return {"control/baseline_accuracy": first["accuracy"], "control/latest_accuracy": last["accuracy"],
            "control/evaluated_optimizer_updates": last["optimizer_updates"], "control/paid_judge_usd": 0}
