# Execute the frozen confirmation through individual native forwards

## Failed calibration gate

The attempted eight-prompt batching matched all8 calibration predictions but
failed the preregistered score checks. Maximum correct-choice log probability
error was0.4015 nats. Maximum deletion-effect error was0.4796 nats. Identity
and router-versus-output deletion checks were exact. The worker aborted before
any confirmation question. Preserve this failed calibration and its allocation.
The cause has not been isolated between padding, batching and numeric kernels.
Do not describe it as a demonstrated padding bug or relax the tolerance.

## Exact repair

Use `native_expert_confirmation.py`, the original one-prompt native API path:
no padding, no attention-mask argument, and `logits_to_keep=1` on the outer
model forward. The model, prompts,64+64 frozen question IDs, alias scoring,
16 conditions, four primary targets and12 corrected primary intervals remain
unchanged. The final specification is
`evidence/native_expert_confirmation_spec.json`.

Run answer-order seeds0,1,2 in three sequential single-GPU jobs. Do not inspect
test efficacy until all three are complete. Monitor progress and inference
validity only. Every question remains part of the same preregistered test.
The separate replication cohort remains unread. The existing identity and
native deletion equivalence checks run at the start of each seed.

## Budget and outputs

Each job uses one MI210 on AMD devel for at most30 minutes. Reserve all90
GPU-minutes now and submit one allocation at a time. Replace each reservation
with observed allocation after completion. The total cap remains300 minutes.
Preserve and privately archive each seed separately. Combine seeds within
question before the original bootstrap analysis. All metrics, uncertainty,
plots and raw-example rules are unchanged from `native_expert_confirmation.md`.

## Results

Pending.
