# Experiment: GSM8K positive control for the RL pipeline

Status: approved on 2026-11-26. User: “approved, run it”.
The [approved configuration](gsm8k_control.approved_20261126.json) owns the exact scientific settings.
No GPU inference or training had run at approval.

## TLDR

First reproduce the repository's small-model GSM8K learning recipe. Then test the TCS CISPO/fully-async setting bundle on the same model.
Use Qwen2.5-0.5B-Instruct and deterministic numeric rewards. Cap the two stages at four total AMD node-hours on eight GPUs.
Both stages start from the original model. This is a one-seed diagnostic, not a three-seed performance claim.

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

Only CPU data preparation and reward validation have run.
The [preparation evidence](evidence/gsm8k_control_preparation_20261126.json) records hashes, partitions and the complete formatted example.
GPU inference, optimizer updates, validation changes and learning outcomes are unmeasured.

Planned plots:
- Exact training reward and fixed-validation accuracy versus optimizer update, shown separately.
- Valid-box and capped-response fractions versus update.
- Entropy, reference divergence, trainer/rollout mismatch and gradient norm, one quantity per panel.
- Final-minus-initial validation accuracy by stage, with a 95% problem-bootstrap interval and one raw correct/incorrect example.

## Global takeaways

Not yet measured. The completed TCS run remains unchanged.

## Next steps

1. Review the model choice and four-hour total cap.
2. Freeze the approved configs and validate native arguments before submission.
3. Run the reference, inspect its learning and raw answers, then apply the declared gate for CISPO.
4. Supervise through terminal state, immutable upload and W&B readback.

## Questions

Approve this small-model, one-seed control and its four-hour total eight-GPU cap, or keep Qwen3.5-4B for a baseline-only GSM8K ceiling check first?
