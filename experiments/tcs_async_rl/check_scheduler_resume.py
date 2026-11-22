"""Replay the actual installed scheduler's loading failure on a CPU optimizer stub.

Only the Megatron logging import is replaced. The scheduler class and numerical
methods execute from the supplied, checksum-recorded source without modification.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def check(source_path):
    source = source_path.read_text()
    module = ast.parse(source)
    module.body = [n for n in module.body if not (isinstance(n, ast.ImportFrom) and n.module == "megatron.core.utils")]
    namespace = {"log_single_rank": lambda *a, **kw: None}
    exec(compile(module, str(source_path), "exec"), namespace)
    scheduler_type = namespace["OptimizerParamScheduler"]

    def make(total, use_checkpoint):
        return scheduler_type(
            SimpleNamespace(param_groups=[{"lr_mult": 1.0, "wd_mult": 1.0}]),
            init_lr=0,
            max_lr=1e-6,
            min_lr=0,
            lr_warmup_steps=0,
            lr_decay_steps=total,
            lr_decay_style="constant",
            start_wd=0.1,
            end_wd=0.1,
            wd_incr_steps=total,
            wd_incr_style="constant",
            use_checkpoint_opt_param_scheduler=use_checkpoint,
            override_opt_param_scheduler=False,
        )

    old = make(1280, False)
    old.step(800)
    state = old.state_dict()
    try:
        make(2080, False).load_state_dict(state)
    except AssertionError as error:
        failure = str(error)
        assert "2080" in failure and "1280" in failure
    else:
        raise AssertionError("Negative control must reproduce the actual failure")
    repaired = make(2080, True)
    repaired.load_state_dict(state)
    assert repaired.num_steps == 800
    rates = []
    for _ in range(40):
        repaired.step(32)
        rates.append(repaired.optimizer.param_groups[0]["lr"])
    assert repaired.num_steps == 2080 and all(lr == 1e-6 for lr in rates)
    return {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "negative_control_error": failure,
        "logging_import_only_stubbed": True,
        "restored_num_steps": 800,
        "final_num_steps": repaired.num_steps,
        "additional_optimizer_steps": len(rates),
        "learning_rates": sorted(set(rates)),
        "weight_decay": repaired.optimizer.param_groups[0]["weight_decay"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    result = check(cli.scheduler_source)
    cli.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
