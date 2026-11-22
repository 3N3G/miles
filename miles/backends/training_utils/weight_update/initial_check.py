"""Verify initial broadcasts against the appropriate pretraining or resumed baseline."""

import logging
from argparse import Namespace
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


async def check_initial_weight_update(
    args: Namespace, *, inference_controller, republish_weights: Callable[[], Awaitable[None]]
) -> None:
    if not args.check_weight_update_equal:
        return
    options = dict(
        allow_quant_error=args.check_weight_update_allow_quant_error,
        selector=args.check_weight_update_selector,
        skip_list=args.check_weight_update_skip_list,
    )
    if args.start_rollout_id > 0:
        # Startup captured the HF initialization. A trained checkpoint differs
        # from that baseline. Check a second transfer of the restored weights.
        logger.info("Checking restored weight broadcast at rollout %d", args.start_rollout_id)
        await inference_controller.check_weights(action="snapshot", **options)
        await inference_controller.check_weights(action="reset_tensors", **options)
        await republish_weights()
    await inference_controller.check_weights(action="compare", **options)
