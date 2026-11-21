# Retrospective sampler-entropy estimate

The seven repeated questions have mean sampled surprisal **0.43747 → 0.44422 nats per token**.
Three question means increase and four decrease. This estimate does not show a large fall across those visits.

The calculation reuses the 864 audited answers from the 27-update Qwen3.5-4B easy-only attempt.
It averages negative sampled-token log-probability within each answer, then weights answers equally within questions.
The probabilities include the original top-p/top-k filter. This estimates behavior-sampler uncertainty on generated prefixes.
It does not recover full-policy entropy or initial-policy divergence. No new model, judge or held-out data was used.

One seed-selected question asks for a spanning-tree packing strategy that bounds how often a tree crosses a prescribed cut.
Its group estimate changes 0.46937 → 0.45488 nats, and reward changes 5.90625 → 6.25 out of 7.
The preselected raw answers have estimates 0.48157 → 0.47314, with rewards 5 → 6.5.
The private report includes both full inputs, outputs and item judgments next to the aggregates.

These are descriptive training observations from one seed. Only seven questions repeat.
Answer length, generated prefixes and mixed weight versions limit direct comparisons.
The estimate cannot establish reward improvement or rule out changes in full-policy entropy.

Evidence and reproduction: [record](evidence/sampled_entropy_20261121.json).
The missing full-policy measurements are addressed by [the instrumentation repair](DIAGNOSTICS_REPAIR_20261121.md).
