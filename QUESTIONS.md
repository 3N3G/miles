# Open questions

## 2026-11-18: TCS solver and judge token budgets

- Question: do 90k/128k tokens mean the Qwen answer limit or the judge completion limit?
- Options: Qwen at 90,000 tokens, Qwen at 128,000 tokens, or a judge-side limit.
- Recommendation: treat these as candidate solver limits. Keep the judge budget separate.
- Blocks: final context sizes, memory checks, walltime estimates, and a submission-ready plan.
- Status: asked in chat. No limit has been selected.

## 2026-11-18: TCS thinking mode

- Question: should Qwen use thinking mode? Ian's reference command disables it.
- Options: enable thinking, or preserve the reference's disabled mode.
- Recommendation: enable it for the proposed reasoning pilot, subject to the user's decision.
- Blocks: the exact solver prompt and the rule for which answer text the judge sees.
- Status: asked in chat. Both tokenizer template previews are saved locally.

## 2026-11-18: Pilot resource limits

- Question: what GPU-hour and judge-API spending caps should the pilot use?
- Options: a bounded technical smoke test, or a bounded smoke test plus learning pilot.
- Recommendation: approve a short runtime check before a learning allocation.
- Blocks: exact jobs, walltimes, sample counts, judge concurrency, and paid calls.
- Status: asked in chat. No GPU job or paid API call has been submitted.

## 2026-11-18: Development split and reward contract

- Question: approve a paper-level development split and the judge's rubric-scoring contract?
- Options: reserve development papers within the ten eligible cohorts, or use training diagnostics only.
  Grade existing rubric items as binary decisions with their supplied weights, or review another rule.
- Recommendation: reserve development papers. Review a small human-labeled judge packet before training.
  Use a second provider for independent assessment. Do not infer proof correctness from judge agreement.
- Blocks: implementing the experiment's data manifest, judge prompt, parser, reward, and selection rule.
- Status: proposed in [the experiment draft](experiments/tcs_async_rl/PLAN.md). No split or reward is implemented.

# Answered questions

- 2026-11-18: hold out FOCS 2026 and STOC 2026. User's explicit instruction.
  SODA 2026 remains eligible. No held-out Parquet shard was downloaded.
- 2026-11-18: Ian's command is a usage example. Replace settings that do not fit TCS.
  The reference command does not freeze batch sizes, topology, hooks, or experiment-specific settings.
- 2026-11-18: create a user-owned Miles fork. User said, "github yea lets fork."
  Creation is authorized. GitHub authentication is still missing on the VM and AMD.
  The available GitHub integration is not installed/connected. No fork has been created.
  Keep private dataset examples outside the public source repository.
