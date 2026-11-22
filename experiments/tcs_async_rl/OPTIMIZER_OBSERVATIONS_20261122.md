# Scope of the logged gradient norm

The installed optimizer clips the main policy and multi-token prediction (MTP) parameters separately.
With `mtp_detach_heads=true`, the model tags MTP parameters with `grad_norm_group='mtp'`.
The main norm excludes that group. The optimizer already computes a separate MTP norm during each update.

This clarifies the previous all-zero-reward group: its logged main-policy norm is zero,
while the reported MTP loss is 0.63560. Zero main-policy norm does not establish zero MTP gradients.
Loss magnitudes alone do not establish that MTP dominates the policy update.
The separate clipping groups rule out competition for one shared clipping threshold under this configuration.
These source checks do not identify why rewards have not consistently increased.

The read-only [optimizer probe](optimizer_probe.py) reads cached component norms, learning rates and scheduler position.
It performs no forward/backward pass, gradient collective, parameter read or optimizer update.
The observer's CPU test verifies that it does not call a gradient-norm method or change optimizer fields.
Native Ray access still needs qualification when the replacement allocation starts.
Use it alongside full/sampling entropy, sampled initial-policy drift and fixed-batch update direction.

Source locations in the pinned runtime:

- `megatron/core/optimizer/optimizer.py`: `get_grads_for_grad_norm`, `_compute_grad_norms_by_group`, `clip_grad_norm`.
- `megatron/core/transformer/multi_token_prediction.py`: parameter tagging under `mtp_detach_heads`.
- `miles/backends/megatron_utils/model.py`: the main norm passed to `log_train_step`.

The runtime image and original numerical evidence are pinned by [the approved plan](EASY_CONTINUATION_PLAN_20261122_APPROVED.md)
and [the full-metric audit](RL_NOISE_AUDIT_20261121.md).
