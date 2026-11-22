# Easy-question reward and optimizer audit

The reward chart mixes different questions. An offline W&B merge also misassigned some episode-reward coordinates.
Corrected data show no consistent increase: seven repeated questions average **4.464 → 4.536 out of 7**.
Four means increase and three decrease. Thirteen questions have only one observed training visit.

The attempt trained Qwen3.5-4B with thinking, one question and 32 answers per update.
It completed 27 updates within four hours, covering 20 questions and 864 answers.
Luna medium graded the supplied rubrics in both anonymous answer orders.
These are provisional training observations from one seed, without a frozen-policy comparison.

- Group means range from 0 to 6.875. Raw differences between successive questions are not learning progress.
- Main-policy gradients are finite. Nonzero norms range from 0.173 to 0.607, below the clipping threshold of one.
  One constant-zero group has zero new policy gradient. No update is skipped. CISPO clipping is zero throughout.
- All three rollout engines match at 28 weight publications. Maximum observed decode lag is one update.
  Per-token effective sample-size ratios stay above 0.9992. These do not measure drift from initialization.
- Candidate scores differ between the two judge calls for 215/864 answers (24.9%).
  The maximum difference is four points. Human calibration and independent assessment remain absent.
- Generation averages 34,573 tokens per answer. Actor waiting accounts for 76% of summed step time.
  Admission permits one group in flight. Visible decode logs come from one engine actor.

The audit checked every consumed answer against both judge receipts.
All 29.87 million response-token positions passed log-probability, loss-mask, sampled-support and version checks.
The private report links a seed-selected raw answer and item-level judgments for every group.
The constant-zero group's example confuses an extremal bound with a random-process guarantee.

The evidence does not identify an optimal learning rate or isolate a cause of absent improvement.
Entropy, cumulative reference-policy divergence and component gradient norms were not measured.
The installed optimizer clips MTP parameters separately. The reported norm excludes them.
See [the source-backed clarification](OPTIMIZER_OBSERVATIONS_20261122.md).
The [evidence record](evidence/easy_overfit_noise_audit_20261121.json) links the full report, all 75 native scalar metrics,
paired-question plots, raw examples and reproduction details.
Use the [corrected W&B series](WANDB.md#verify-offline-metric-axes). Original grades and used configs remain unchanged.
