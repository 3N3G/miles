"""Read back every native training scalar and fixed GSM8K evaluation from W&B."""

import argparse
import json
from pathlib import Path

from experiments.tcs_async_rl.gsm8k_artifacts import write_json
from experiments.tcs_async_rl.native_metrics import read_native_metrics


def verify(directory):
    complete = json.loads((directory / "control-complete.json").read_text())
    monitor = json.loads((directory / "monitor-complete.json").read_text())
    if monitor["scheduler"]["state"] != "COMPLETED":
        raise ValueError("Final verification requires a completed Slurm job")
    # The SDK is an optional VM reporting dependency. AMD needs no credentials.
    import wandb

    config = json.loads((directory / "resolved-config.json").read_text())
    args = config["args"]
    path = f"{args['wandb_team']}/{args['wandb_project']}/{config['run_id']}-native"
    run = wandb.Api().run(path)
    history = list(run.scan_history(use_cache=False))
    native = read_native_metrics(directory, allow_pending=True, require_publication=True)
    if len(native) != complete["optimizer_updates"]:
        raise ValueError("Native optimizer history differs from the final checkpoint boundary")
    training = [row for row in history if row.get("audit/train/pg_loss") is not None]
    observed = {}
    conflicts = []
    for row in training:
        step = row["audit/optimizer_step"]
        for key, value in row.items():
            if not key.startswith("audit/") or value is None:
                continue
            coordinate = (step, key)
            if coordinate in observed and observed[coordinate] != value:
                conflicts.append({"step": step, "metric": key})
            observed[coordinate] = value
    missing, differing, checked = [], [], 0
    for row in native:
        step = row["optimizer_step"]
        for metric, value in row.items():
            key = "audit/" + metric
            coordinate = (step, key)
            if coordinate not in observed:
                missing.append({"step": step, "metric": key})
            elif observed[coordinate] != value:
                differing.append({"step": step, "metric": key, "native": value, "wandb": observed[coordinate]})
            checked += 1
    evaluation_rows = [row for row in history if row.get("control/eval/accuracy") is not None]
    evaluation_errors = []
    evaluation_scalars = 0
    for file in sorted((directory / "evaluations").glob("*.json")):
        point = json.loads(file.read_text())
        matches = [row for row in evaluation_rows if row.get("control/eval_optimizer_updates") == point["optimizer_updates"]]
        for metric in ("accuracy", "valid_box_fraction", "capped_fraction", "mean_response_tokens"):
            values = [row.get("control/eval/" + metric) for row in matches]
            if not values or any(value != point[metric] for value in values):
                evaluation_errors.append({"updates": point["optimizer_updates"], "metric": metric, "observed": values})
            evaluation_scalars += 1
    result = {
        "url": run.url,
        "state": run.state,
        "history_rows": len(history),
        "native_training_points": len(native),
        "wandb_training_rows": len(training),
        "native_scalars_checked": checked,
        "evaluation_rows": len(evaluation_rows),
        "evaluation_scalars_checked": evaluation_scalars,
        "conflicting_coordinates": conflicts,
        "missing_native_scalars": missing,
        "differing_native_scalars": differing,
        "evaluation_errors": evaluation_errors,
    }
    expected_steps = {row["optimizer_step"] for row in native}
    result["unexpected_training_steps"] = sorted({row["audit/optimizer_step"] for row in training} - expected_steps)
    result["passed"] = run.state == "finished" and not any((conflicts, missing, differing, evaluation_errors, result["unexpected_training_steps"]))
    write_json(directory / "wandb-full-verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    result = verify(parser.parse_args().directory)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)
