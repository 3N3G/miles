# Experiment: GSM8K positive control for the RL pipeline

Status: approved on 2026-11-26. User: “approved, run it”.
The [approved configuration](gsm8k_control.approved_20261126.json) owns the exact scientific settings.
No GPU inference or training had run at approval.

## TLDR

The reference completed 250 updates. Fixed-validation accuracy rose from 172/256 (67.2%) to 185/256 (72.3%).
The paired 95% interval for the change includes zero. This single-seed result passes the predeclared gate, but does not establish repeatable improvement.
The fresh-weight CISPO stage is queued. No CISPO inference or training has run. Its original eight-MI350 setup remains authorized.
The approved cap remains four total AMD node-hours on eight GPUs, with no paid judge.

## Background / motivation

The TCS continuation produced no matched-question score gain. Its proof judge, long outputs and hybrid model complicate diagnosis.
GSM8K removes the learned judge. A small model gives a clearer opportunity to learn than a model already near the accuracy ceiling.
No benchmark guarantees improvement for every model and configuration.

## Concrete question we are answering

Can the existing reference recipe and our CISPO/fully-async setting bundle improve exact numeric accuracy on GSM8K?
If both work, the small-model pipeline can learn with exact rewards. This does not validate Qwen3.5 kernels, 90k-token learning or TCS grading.
If only the reference works, investigate the changed setting bundle. Several settings differ, so this does not identify CISPO alone.
If the reference fails, diagnose it before allocating the second stage. A crash, insufficient updates and finite flat learning are distinct outcomes.

The cheapest checks already pass: reward controls on every source training row and a hand-read formatted example.
Before training, measure initial validation accuracy and check for varied training rewards. Stop as uninformative at a 95% ceiling or no varied groups.

## Methodology

- Models: Qwen2.5-0.5B-Instruct, pinned in the configuration. Fresh initial weights for each stage. No TCS checkpoint.
- Data: the GSM8K preparation used by the repository's learning test. Split its 7,473 training rows into 7,217 training and 256 validation rows with a fixed seed. Do not read the official test split.
- Concrete example: Natalia sells 48 clips, then half as many. The formatted input contains the question and a request for a boxed answer. The answer is 72. The grader gives boxed 72 one point and boxed 73 zero.
- Reward: native `--rm-type math`, which extracts the boxed answer and checks numeric/symbolic equality. No API judge or format bonus.
- Reference stage: the checked-in GRPO clipped-PPO async recipe, with 32 questions ×8 answers, batch 256 and at most 250 updates.
- CISPO stage: the TCS setting bundle, with four questions ×32 answers, batch 128, actual rollout probabilities, support replay and fully-async generation. At most 250 updates. Batch, filtering, sampling and scheduling differences are explicit in the configuration.
- Shared optimizer: Adam, learning rate 1e-6, betas 0.9/0.98 and weight decay 0.1. Responses are capped at 1,024 tokens.
- Machine changes: two actor GPUs and six single-GPU rollout engines. The reference test uses one actor and one rollout GPU. AMD reports an exclusive eight-GPU partition, so the cap counts all eight GPUs.
- Budget: at most two node-hours per stage, four combined, including setup and finalization. Stop after the update limit if earlier. Paid judge cost is zero.
- Seed: 20261126. This initial diagnostic deliberately uses one training seed. Replication across three or more seeds needs a follow-up decision.
- Primary metric: final-minus-initial exact accuracy on the fixed validation set. Use the final checkpoint, not the best intermediate point.
- Controls: correct references, deliberately changed numeric answers, empty and irrelevant responses. All 7,473 examples pass each CPU check.
- Statistics: exact paired McNemar tests with Bonferroni correction for two planned comparisons. Report a 95% paired problem-bootstrap interval. This measures question variation, not training-seed variation.
- Success gate: positive final-minus-initial validation accuracy and a correctly directed local optimizer update. Report uncertainty and significance separately. Never call a flat high-baseline curve a learning success.
- Logging: exact accuracy, valid-box fraction, cap fraction, constant groups, entropy, reference divergence, rollout mismatch, importance weights, gradients, accumulation, weight versions and update direction. Compare W&B coordinates against native events.
- Records: preserve raw outputs and numeric verdicts in the private HF run archive. Preserve used configs. Keep checkpoints on AMD. Use one serialized VM W&B writer.

The reference is [the local learning test](../../tests/e2e/long/test_qwen2.5_0.5B_gsm8k_async.py).
Its 55% test threshold is a declared CI condition, not a measured result for our current image or validation split.
Sources: [GSM8K](https://github.com/openai/grade-school-math) and [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

## Results

The reference completed 250 optimizer updates and consumed 64,000 answers. Final fixed-validation accuracy was 185/256 (72.27%), versus 172/256 (67.19%) initially.
Thirty questions changed from wrong to right, while 17 changed from right to wrong. The gain was 5.08 percentage points.
The 95% paired question-bootstrap interval was 0–10.16 points. The exact paired McNemar p-value was 0.07894, or 0.15788 after the planned two-comparison correction.
This uncertainty concerns validation questions. One training seed does not measure variation across training seeds.
The final checkpoint supplied the primary result, even though an intermediate checkpoint scored higher.

Formatting explains some improvement. A seeded example reasoned correctly that four bottles need 4 × 85 − 33 = 307 stars, but omitted the required box before training.
The final answer boxed 307 and earned one point. Among 252 responses with boxes at both stages, correct answers increased from 172 to 182.
A seeded regression changed a correct zoo total of 144 to 108 after using the wrong giraffe-enclosure count.
Full paired examples are in `paired-raw-answer-audit.json` in the archive below.

All three local optimizer-direction checks were positive. Gradients stayed finite, and six rollout engines agreed on each of 250 updated weight publications.
The saved iteration 249 matches 250 completed updates. W&B matched all 18,029 native training scalars and 56 validation scalars.
Baseline/final valid-box fractions were 253/256 and 255/256. Token caps affected zero and one validation answer respectively.
The official GSM8K test split remains unread.
An [independent arithmetic audit](evidence/gsm8k_numeric_audit_20261127.json) matched 3,556 parsed numeric values across all 14 validation checkpoints, with zero disagreements.
It also checked 18 missing-box zeros. Ten symbolic or nonscalar boxes remain outside that checker. This does not independently validate box extraction or benchmark labels.

The reference used 3,398 allocated node-seconds across attempts, including 319 seconds in two startup failures. Judge cost was zero.
Its [immutable archive](https://huggingface.co/datasets/3N3G/internalization-runs/tree/318eecb525cdd1606602ac62facda521b2313ead/miles/gsm8k-reference-v4)
contains 596 verified artifacts. Its [W&B run](https://forge.coreweave.com/wandb/geney/tcs-async-rl/runs/gsm8k-reference-261127-090520-021-native) is finished.
The [machine-readable result](evidence/gsm8k_reference_final_20261127.json) owns the exact measurements.

CISPO job 562347 remains queued with zero allocated seconds and zero updates. The [resource choice](../../QUESTIONS.md) remains open.
This positive 0.5B reference result does not validate the original TCS CISPO bundle, Qwen3.5 kernels, long responses, or the proof judge.

Plots (reference complete, CISPO pending):
- Exact training reward and fixed-validation accuracy versus optimizer update, shown separately.
- Valid-box and capped-response fractions versus update.
- Entropy, reference divergence, trainer/rollout mismatch and gradient norm, one quantity per panel.
- Final-minus-initial validation accuracy by stage, with a 95% problem-bootstrap interval and one raw correct/incorrect example.

## Global takeaways

Interpretation remains open. The measured reference gain passes the declared gate. The CISPO stage has not run.

## Next steps

1. Keep the approved CISPO job queued and supervised while the resource-variant question is pending.
2. Run CISPO from fresh original weights, using either the original allocation or an explicitly approved resource variant.
3. Inspect its final checkpoint, matched-question accuracy, optimizer diagnostics and raw answers.
4. Verify all W&B scalars and the immutable result archive before reporting completion.

## Questions

The small-model, one-seed control and four-hour total eight-GPU cap were approved. The original resource plan remains authorized.
A smaller-resource variant needs the user's decision because it changes scheduling or hardware. See [the recorded choices](gsm8k_cispo_resources.proposed_20261127.json).
