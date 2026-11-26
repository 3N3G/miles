# Audit of the easy and medium RL continuation

The fixed-question comparison is still pending. The mixed-question training curve cannot establish improvement or failure.
The checks below identify limitations to test. They do not establish a single cause.
The running experiment keeps its approved settings and budget.

The optimizer changes probabilities in the intended direction on saved batches.
The strongest current concerns are long unfinished answers, few questions per update, loss normalization, and reward validity.
Luna usually distinguishes a completed proof from a failed attempt in the inspected cases.
Its partial score does not mean that the theorem was proved.

A fresh check hid Luna's scores until eight assistant item assessments were saved.
It matched **6/8 totals and 38/40 items**. All eight totals fell within the recorded uncertainty ranges.
No inspected full-credit answer had a clear substantive error.
One disagreement concerns an implicit feasibility argument. Another concerns an elementary finiteness consequence.
These are assistant assessments, not independent human calibration.
The [frozen labels and original decisions](evidence/long236_fresh_judge_audit_20261126.json) preserve both disagreements.

The next decisive measurement uses the same twelve questions before and after training.
Its baseline is checkpoint108, which already received easy-question RL. It is not the original pretrained model.
One training seed and twelve training-eligible questions cannot establish held-out generalization.

## Comparison with primary guides

| Component | What the sources say | What this run does | Audit finding |
|---|---|---|---|
| CISPO objective | The original paper clips detached importance weights and uses token aggregation. | The detached weight and gradient direction match. The reducer averages tokens within each answer, then answers. | A real normalization difference. It changes the relative weight of long answers. It is not evidence of a sign bug. |
| Cutoff handling | DAPO and TRL describe noisy penalties on unfinished responses and masking as one remedy. | Every token in the recent capped answers remains active. The judge scores only the extracted final answer. | An unfinished reasoning trace can receive zero for an empty final. This can suppress useful prefixes as well as failed reasoning. |
| Group normalization | Dr. GRPO and current guides discuss bias from response-length and group-standard-deviation normalization. | Both per-answer length division and within-question standard-deviation scaling are enabled. | Another credible weighting issue. Switching normalization requires a controlled experiment. |
| Memory and batch size | Verl separates memory-sized microbatches from the statistical batch. | Four questions ×32 answers per optimizer update. Dynamic packing accumulates many microbatches. | Accumulation is not one. Four independent question clusters still give noisy updates. More asynchronous sampling alone does not enlarge the optimizer batch. |
| Asynchronous correction | Verl requires recorded behavior probabilities and explicit correction for stale/backend-shifted samples. | The loss uses actual rollout log probabilities and replays the recorded sampling support. Token-level CISPO ratios supply correction. | The common wrong-denominator bug is addressed. Token-level effective sample size does not certify whole-answer freshness. |
| Judge validity | Proof-evaluation research separates rubric agreement from expert proof correctness. | A labeled reference, original weighted Boolean items, final-answer-only grading, saved requests and raw reviews. | Controls are useful but weak on subtle incorrect proofs. Independent expert calibration remains absent. |

Sources: [MiniMax-M1, sections3.1,3.2 and5](https://arxiv.org/html/2506.13585v1),
[DAPO, section3.4](https://arxiv.org/html/2503.14476v2),
[TRL GRPO guide](https://huggingface.co/docs/trl/main/en/grpo_trainer),
[verl GRPO guide](https://verl.readthedocs.io/en/latest/algo/grpo.html),
[verl rollout correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html),
[Dr. GRPO](https://arxiv.org/abs/2503.20783),
and [QEDBench](https://arxiv.org/html/2602.20629v2).

## Direct evidence and limits

- Through checkpoint135, the durable continuation contains27 new updates and3,456 answers.
  It consumed108/236 question rows:11/20 easy and97/216 medium.
  Those answers contain240,639,530 generated tokens. Of3,456 answers,1,245 reach the cap (36.0%).
  Eleven of108 question groups have constant rewards and provide no policy-gradient signal.
  Prepared corpus size therefore overstates actual training exposure if these counts are omitted.
- The published update134 has raw return1.695/7, 78/128 caps and57 empty finals.
  Update135 has return3.242/7, 37 caps and25 empty finals.
  These batches use different questions. Their reward difference is not a learning estimate.
- Zero-score answers average88,841 and86,385 tokens in these two batches.
  Positive-score answers average67,901 and55,747 tokens.
  This association is descriptive. Difficulty and stopping behavior are confounded.
- All capped response tokens in these two batches are active in the policy loss.
  Some caps contain useful final text and earn points. Masking every cap would also discard these answers.
- About55.0% and57.6% of sampled token positions have a singleton sampling support.
  Renormalization on that fixed singleton gives zero policy gradient at that position.
  Other positions still provide gradients. This does not establish that sampling controls caused flat returns.
- A same-weight pair of earlier batches changed raw return by2.148 points without a policy update.
  This is direct evidence that question mix can dominate the raw curve.
- Saved update130 increases mean token log probability for positive-advantage answers and decreases it for negative-advantage answers.
  The measured surrogate gain is0.0000684. Local optimizer progress does not imply better mathematical proofs.
- Update134 has finite gradient norm0.0858, reference divergence0.0100 and train/rollout divergence0.000289.
  Importance-weight effective sample size is0.99942, and clipping is zero.
  These are per-token diagnostics. They do not bound the product of90,000 probability ratios.

The recent trace audit verifies all256 answers, original judge requests, score parsing, masks, behavior probabilities,
sampling support and weight-version counts. Both native reward/length means agree exactly.
All256 formatted prompts equal the pinned problem-only Qwen thinking template. They contain neither reference nor rubric.
The raw eight-answer review is separate from the uniform examples used in hourly monitoring.

## Code paths checked

- [CISPO weights](../../miles/backends/training_utils/loss/hub/math_utils.py): `compute_cispo_loss`.
- [Reduction](../../miles/backends/training_utils/data/context_parallel.py): `get_sum_of_sample_mean`.
- [Accumulation scaling](../../miles/backends/training_utils/loss/objective.py): `loss_function`.
- [Reward grouping](../../miles/ray/rollout/train_data_conversion.py): `_normalize_rewards_by_rollout`.
- [Final extraction and transport](reward_hook.py): `scoring_request`.
- [Judge identity and cache](reward_service.py): `Scorer.score` and `_score`.

Native parsed arguments confirm `calculate_per_token_loss=False`, group standard-deviation normalization,
global batch128, microbatch1, no cutoff filter, zero entropy/KL penalties, and BF16 output-head computation.
Entropy and reference divergence are observed even though their optimization penalties are zero.
The vocabulary and padded vocabulary both have248,320 entries.
The cap is90,000 generated tokens, inside a98,304-token context. No prompt approaches that context limit.

The installed Megatron source detaches the main hidden states, token embedding output and output weight from the
multi-token prediction loss. That auxiliary loss therefore does not supply a competing backbone language-modeling objective.
The inspected file has SHA256`f9843d7982bb4deb605537dd53a651e829b38628a614fcfabb93d486c11509bf`.
Relevant lines are1770,2136 and2930. MTP parameters have their own gradient-norm group.

The MiniMax paper reports an FP32 output-head repair for its hybrid architecture.
Our BF16 head remains a diagnostic candidate, rather than a demonstrated failure.
Finite losses and small average token mismatch do not prove exact numerical equivalence over a long trajectory.
An isolated fixed-weight probability comparison would be needed before attributing failure to this issue.

## Judge checks and unresolved reward alignment

The accepted v5 judge matched34/40 saved totals and96.17% of case-balanced items before launch.
It failed the original90% exact-total gate. The user accepted that measured limitation.
The reference/empty/irrelevant corpus controls include472 negative answers, all scored zero.
Four supplied references received less than seven because of reviewed reference gaps.
These controls do not test sensitivity to a plausible proof with one hidden mathematical error.

One inspected matroid answer earns4/7 for separate combinatorial and weight calculations despite failing the matroid axioms.
That follows the approved component-credit rule. It exposes a possible mismatch between maximizing rubric points and proving the theorem.
Another fresh failed online-selection proof receives0/7, correctly identifying its unknown-marginal and adversarial-order errors.
A correct summary-aggregation construction receives7/7 and proves a risk gap of at least1/2.
All three types of evidence matter when assessing the reward.

After the fixed comparison, the audit will distinguish verified implementation faults, measurement confounds,
and research choices. Any proposed new loss, sampling setting, reward or training budget needs a separate reviewed plan.
