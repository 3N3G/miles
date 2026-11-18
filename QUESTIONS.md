# Open questions

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
- 2026-11-18: the user created `3N3G/miles`. GitHub metadata confirms its parent is `IanYHWu/miles`.
  Use SSH for VM writes and HTTPS for AMD reads. The user registered the VM's dedicated deploy key.
  SSH read/write access is verified. Preparation commits are pushed and the AMD checkout is synchronized.
  Keep private dataset examples outside the public source repository.
- 2026-11-18: use a 90,000-token solver output cap. User said, "yeah lets do 90k."
  Use the matching planned total context of 98,304 tokens. The judge budget remains separate.
  The 90k option came from Ian's quoted suggestion, not measured Qwen generation lengths.
  This token-budget decision does not authorize GPU submissions or paid judge calls.
