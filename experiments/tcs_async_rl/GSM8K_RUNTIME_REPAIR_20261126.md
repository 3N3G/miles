# Restore reference prompt shuffling before allocation

The approved control calls for the repository GSM8K reference recipe. Its command enables `--rollout-shuffle`.
The submitted wrapper omitted that setting. Native parsing defaults to false. The exported training rows retain source order.
This was found before job561051 received an allocation. No inference or optimizer update ran.

The repair restores `rollout_shuffle=true` for both control arms. The native rollout shuffle seed remains42, as in the source recipe.
The original approved config is preserved byte-for-byte in `gsm8k_control.approved_20261126.before_shuffle_repair.json`.
Model weights, dataset membership, prompts, rewards, optimizer settings, validation, update limits and budget caps remain unchanged.
This restores the approved reference behavior. It does not add another experimental condition.

Validation:30 CPU cases pass, including both-arm recipe flags and final optimizer-prefix checks.
GPU runtime validation remains pending. Preserve the old run directory. Cancel the still-pending own job and submit the repaired reference in a fresh directory.
The full reference allowance is still at most7200 allocated seconds across all attempts. CISPO retains its independent7200-second cap and the same learning gate.
Every submission and terminal state must be recorded. Supervision, final checkpoint/evaluation matching, W&B readback and immutable result upload remain required.
