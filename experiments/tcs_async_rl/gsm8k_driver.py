"""Delegate native async training; add baseline gating and final evaluation only."""

import asyncio
import json
from pathlib import Path

from experiments.tcs_async_rl.gsm8k_artifacts import export_evaluation, write_json


class UninformativeControl(RuntimeError):
    pass


async def train(args, *, disposer):
    # Keep distributed imports out of CPU checks. No optimizer code is replaced.
    import train_async as native

    root = Path(args.save_debug_rollout_data).parent.parent
    state = {"last_step": None, "last_eval_updates": None, "controller": None, "dispatcher": None}
    original_update, original_dispatcher = native.update_weights, native.EvalDispatcher

    async def tracked_update(*positional, **keywords):
        result = await original_update(*positional, **keywords)
        state["controller"] = positional[3]
        if keywords.get("rollout_id") is not None:
            state["last_step"] = keywords["rollout_id"]
        return result

    class RecordedDispatcher(original_dispatcher):
        def __init__(self, *positional, **keywords):
            super().__init__(*positional, **keywords)
            state["dispatcher"] = self

        async def dispatch(self, rollout_id, hf_dir=None, force=False):
            await super().dispatch(rollout_id, hf_dir=hf_dir, force=force)
            updates = 0 if state["last_step"] is None else state["last_step"] + 1
            result = export_evaluation(args, rollout_id=rollout_id, optimizer_updates=updates)
            state["last_eval_updates"] = updates
            if updates == 0 and result["accuracy"] >= 0.95:
                write_json(root / "uninformative.json", {"reason": "Baseline accuracy at least 95%", **result})
                raise UninformativeControl("Baseline is near ceiling")

    native.update_weights, native.EvalDispatcher = tracked_update, RecordedDispatcher
    try:
        await native.train(args, disposer=disposer)
        updates = 0 if state["last_step"] is None else state["last_step"] + 1
        if updates and state["last_eval_updates"] != updates:
            await state["controller"].prepare_eval()
            await state["dispatcher"].dispatch(state["last_step"], force=True)
        result = {"optimizer_updates": updates, "completed": True, "uninformative": False}
    except Exception as error:
        expected = isinstance(error, UninformativeControl) or "Approved control has no reward variation" in str(error)
        if not expected or not (root / "uninformative.json").exists():
            raise
        updates = 0 if state["last_step"] is None else state["last_step"] + 1
        result = {"completed": True, "uninformative": True, "optimizer_updates": updates}
    finally:
        native.update_weights, native.EvalDispatcher = original_update, original_dispatcher
    write_json(root / "control-complete.json", result)


if __name__ == "__main__":
    from miles.utils.arguments import parse_args
    from miles.utils.async_utils import with_disposer

    asyncio.run(with_disposer(train, parse_args()))
