"""Evaluate a restored checkpoint on a fixed dataset without an optimizer update."""

import asyncio
import collections
import gzip
import json
from functools import partial
from pathlib import Path

from experiments.tcs_async_rl.rl_artifacts import _atomic_bytes, file_sha256, portable_sample


def verify_publication(source, current, *, rollout_id, restored_events, expected_publications):
    def publications(data):
        return [
            row
            for line in data.splitlines()
            if (row := json.loads(line))["type"] == "inference_engine_weight_checksum"
        ]

    old = [row for row in publications(source.read_bytes()) if row["rollout_id"] == rollout_id]
    inherited, actual = restored_events.read_bytes(), current.read_bytes()
    if not actual.startswith(inherited):
        raise ValueError("Screen event history differs from the restored checkpoint snapshot")
    new = publications(actual[len(inherited) :])
    if not old or len(new) != expected_publications or any(row["rollout_id"] != rollout_id for row in new):
        raise ValueError("Unexpected restored policy publication count or identity")
    expected = old[-1]["engine_checksums"][0]
    if len(old[-1]["engine_checksums"]) != 3 or any(engine != expected for engine in old[-1]["engine_checksums"]):
        raise ValueError("Saved source engines disagree")
    if any(
        len(row["engine_checksums"]) != 3 or any(engine != expected for engine in row["engine_checksums"])
        for row in new
    ):
        raise ValueError("Screen engines differ from the saved source policy")
    return {
        "rollout_id": rollout_id,
        "engines": 3,
        "matches_source": True,
        "checked_tensor_records_per_engine": len(expected),
        "new_publications_verified": len(new),
        "restored_event_prefix_bytes": len(inherited),
    }


def export_screen(args):
    # Only the GPU allocation reads its own trusted torch snapshot.
    import torch

    root = Path(args.save_debug_rollout_data).parent.parent
    resolved = json.loads((root / "resolved-config.json").read_text())
    snapshot_path = Path(args.save_debug_rollout_data.format(rollout_id=f"eval_{args.start_rollout_id - 1}"))
    snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    samples = snapshot["samples"]
    questions = json.loads((root / "input.json").read_text())["questions"]
    counts = collections.Counter(sample["metadata"]["question_key"] for sample in samples)
    from experiments.tcs_async_rl.reward_hook import question_key

    expected = {question_key(row): args.n_samples_per_eval_prompt for row in questions}
    if dict(counts) != expected or len({sample["index"] for sample in samples}) != len(samples):
        raise ValueError("Screen did not generate the exact pinned question and answer counts")
    target = snapshot_path.with_suffix(".jsonl.gz")
    with gzip.open(target.with_suffix(".partial"), "wt", compresslevel=1) as stream:
        for sample in samples:
            stream.write(json.dumps(portable_sample(sample), ensure_ascii=False, allow_nan=False) + "\n")
    target.with_suffix(".partial").replace(target)
    result = {
        "optimizer_updates": 0,
        "checkpoint_rollout_id": args.start_rollout_id - 1,
        "answers": len(samples),
        "questions": len(counts),
        "samples_per_question": args.n_samples_per_eval_prompt,
        "dataset_sha256": resolved["dataset_sha256"],
        "responses": str(target),
        "responses_sha256": file_sha256(target),
        "publication": resolved["source_publication_check"],
    }
    _atomic_bytes(root / "screen-complete.json", (json.dumps(result, indent=2) + "\n").encode())
    snapshot_path.unlink()  # The verified gzip preserves this run's complete inference snapshot.


async def screen(args, *, disposer):
    # Defer distributed dependencies so pure publication checks work on the VM.
    from miles.backends.training_utils.weight_update.initial_check import check_initial_weight_update
    from miles.ray.placement_group import create_rollout_components, create_training_models, update_weights
    from miles.utils.orchestration_utils import init_orchestration_script

    if not args.reset_rollout_state or args.group_rm or not args.load or args.debug_rollout_only:
        raise ValueError("Checkpoint screening needs real restored trainers, fresh queues and per-answer rewards")
    init_orchestration_script(args, disposer=disposer)
    controller, executor, _ = await create_rollout_components(args)
    disposer.add(controller, executor)
    actor, critic = await create_training_models(args, executor)
    disposer.add(critic, actor)
    await update_weights(args, actor, executor, controller)
    await check_initial_weight_update(
        args,
        inference_controller=controller,
        republish_weights=partial(update_weights, args, actor, executor, controller),
    )
    root = Path(args.save_debug_rollout_data).parent.parent
    resolved = json.loads((root / "resolved-config.json").read_text())
    resolved["source_publication_check"] = verify_publication(
        Path(resolved["source_publication_file"]),
        root / "events/main.jsonl",
        rollout_id=args.start_rollout_id - 1,
        restored_events=Path(args.load) / "rollout" / str(args.start_rollout_id - 1) / "debug_events/main.jsonl",
        expected_publications=2 if args.check_weight_update_equal else 1,
    )
    _atomic_bytes(root / "resolved-config.json", (json.dumps(resolved, indent=2) + "\n").encode())
    await controller.prepare_eval()
    await executor.eval(args.start_rollout_id - 1)
    export_screen(args)


if __name__ == "__main__":
    from miles.utils.arguments import parse_args
    from miles.utils.async_utils import with_disposer

    asyncio.run(with_disposer(screen, parse_args()))
