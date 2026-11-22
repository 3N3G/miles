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
Native Ray access passed on both workers in the replacement allocation.
Before the first new update, both report scheduler counter 800 and learning rate 1e-6.
The cached component norms are absent before backward computation, as expected.
The read-only client needs host PID visibility because Ray 2.58 discovers the raylet through process IDs.
Its first attempt in a separate PID namespace failed. The corrected client only connects, reads and disconnects.
The training job keeps its original PID isolation. No launcher or Ray cleanup command runs in the observer.
After global update 27, both workers report counter 896 and a cached MTP gradient norm of 0.084576.
That update's logged main-policy norm is 0.186181. Both are below their clipping threshold of 1.
These are different parameter groups. Their scalar norms alone do not measure their effects on answer quality.
See [the first native diagnostics](evidence/first_native_rl_diagnostics_20261122.json).
Use it alongside full/sampling entropy, sampled initial-policy drift and fixed-batch update direction.

Source locations in the pinned runtime:

- `megatron/core/optimizer/optimizer.py`: `get_grads_for_grad_norm`, `_compute_grad_norms_by_group`, `clip_grad_norm`.
- `megatron/core/transformer/multi_token_prediction.py`: parameter tagging under `mtp_detach_heads`.
- `miles/backends/megatron_utils/model.py`: the main norm passed to `log_train_step`.

The runtime image and original numerical evidence are pinned by [the approved plan](EASY_CONTINUATION_PLAN_20261122_APPROVED.md)
and [the full-metric audit](RL_NOISE_AUDIT_20261121.md).
