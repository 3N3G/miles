# Repair missing RL observations

The previous attempt did not measure policy entropy or drift from the initial model.
The entropy flag was disabled. Its logged zero was a placeholder.
Its divergence metrics compared the trainer with recent rollout weights.
They could not show cumulative policy drift.

The repair adds three observations without changing rewards or optimizer gradients:

- Full-policy entropy and entropy within the sampler's allowed token set, both in nats per token.
- Sampled divergence from the frozen initial policy, using unfiltered actor and reference scores.
- Scores before and after an update on the exact same batch, to check the policy-gradient direction.

The distinction between entropies matters with top-p and top-k sampling.
For control logits `[1, 0.2, -0.5, 0.4]`, full-policy entropy is 1.258827 nats.
A sampler that permits only one token has zero entropy on that same position.
Identical actor and reference weights give zero divergence in this control.
The existing full-policy entropy contract remains unchanged.
The added reference score is detached and the KL coefficient stays zero.

Ninety TCS tests and 19 CPU loss/diagnostic tests passed.
These include unchanged CISPO losses and gradients, favorable/reversed/zero update controls,
and a two-process check where one tensor-parallel rank has no allowed sampling tokens.
Live W&B controls verified coordinates 25 and 26, running state and finished state.
Separate upload and readback acknowledgements prevent successful uploads from being resent during history-index delays.
The [W&B procedure](WANDB.md#publish-new-runs-from-one-live-writer) owns the logging details.

No GPU job or new training answer was produced during this repair.
The real ROCm reference pass, observation memory cost and checkpoint resume still need GPU qualification.
Sampled divergence on filtered rollout prefixes is not exact full-vocabulary KL.
Correct update direction is a software check. It does not establish better newly generated answers.
The [continuation plan](EASY_CONTINUATION_PLAN_20261121.md) retains the same scientific setup.
Its next allocation and proposed concurrency change await the user's decision.

Operational evidence: [repair checks](evidence/diagnostics_repair_20261121.json).
