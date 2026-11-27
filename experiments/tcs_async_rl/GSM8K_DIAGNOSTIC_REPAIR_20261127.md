# Repair GSM8K startup diagnostics without changing training

Job561193 failed after37 allocated seconds. It produced no inference answers or optimizer updates.
Its frozen source was29fdfcb9236e7205065f01b9b78115ff73e5080c.
Native argument validation rejected the added update-direction observation. That observer previously required one tensor-parallel group and rollout-engine log-probabilities.
The approved control uses two ordinary data-parallel actor ranks. The reference computes old probabilities with its actor forward pass.

The repair uses the probability denominator already selected by the training arguments. It gathers sample-weighted direction statistics across the ordinary data-parallel group.
Tensor-parallel source ranks perform the gather. Only the original main rank logs the combined result. Empty masked shards contribute zero samples.
Context or pipeline parallel observation and independent data parallel training remain unsupported.
The loss, optimizer, model, batches, prompts, numeric reward, fixed validation and both arm settings remain unchanged.
The original configuration bytes remain in Git and the failed run archive. This is a read-only instrumentation repair.

Code inspection also found a pending metadata error: PyTorch argument types require string serialization. Only the validated-argument snapshot enables that conversion.
Metric and answer JSON still reject unsupported values and nonfinite numbers.

Validation:34 CPU cases pass with PyTorch2.10.0+cpu. They include two-process Gloo aggregation with unequal shard counts, a fully masked shard, both approved argument combinations, gradient detachment, and dtype serialization.
Targeted Ruff checks and the whitespace check pass. Actual GPU argument validation and training remain pending.

Retry the reference from the same original HF weights in a fresh run directory. The new allocation requests7140 seconds, rounded down from7163 remaining seconds.
Across attempts the maximum is37+7140=7177 seconds, below the7200-second arm cap. Request save15 minutes before the new deadline. The250-update target and CISPO learning gate remain unchanged.
CISPO still has its separate7200-second cap. No paid judge calls are added.

For this1024-token control, require3 GiB free on the VM before resubmission. Keep the generic long-TCS5 GiB guard unchanged.
Archive portable rollout copies once they total64 MiB. Verify immutable HF hashes before eviction. Keep AMD originals and evaluation outputs.
Return to foreground diagnosis before a sync if VM free space falls below2 GiB. The foreground watcher continues through terminal state and archive/W&B verification.
