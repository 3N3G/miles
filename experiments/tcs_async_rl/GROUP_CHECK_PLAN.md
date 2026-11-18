# Experiment: first TCS inference group on AMD

Status: approved by the user on 2026-11-18: “run a quick inference check of the group first as an amd job.”
The user also approved local VM credential loading for grading. No key values enter AMD or run artifacts.

## TLDR

Generate 32 responses to one previously seeded eligible TCS question using frozen Qwen3.5-4B.
Use thinking and the approved 90,000-token output ceiling. Perform no learning or final-test evaluation.
Grade the completed final answers with the supplied rubric and Luna at medium effort.
These provisional grades test whether this group has score variation. They do not establish judge accuracy or population performance.

## Background / motivation

GRPO uses relative rewards within each prompt's response group. Uniform scores supply no reward-derived advantage.
We have no measured group rewards for this model and dataset. Check generation and grading before allocating training.
The original two-GPU production engine remains unqualified. This bounded diagnostic uses one GPU and 32 running requests.

## Concrete question we are answering

Does this frozen-model group produce complete final answers with varied rubric grades?
If yes, inspect the raw examples and grading controls before proposing a larger baseline or learning run.
If no, distinguish incomplete thinking, transport/parser failure, judge-control failure, and actual uniform scores.
Do not silently alter the model, rubric, prompts, or dataset to manufacture a positive result.
This single question cannot establish a rate across TCS questions.

## Methodology

- Model and dataset revisions: pinned in `config.proposed.json`, frozen with the run commit.
- Data: the eligible example previously selected with seed 20261118. It is SODA 2023, paper
  `Online_Sorting_and_Translational_Packing_of_Convex_Polygons`, question `node_25`, a hard proof-writing task.
  It asks for constructive packing guarantees, including proved strip, square, bin, and perimeter bounds.
  Its six weighted rubric items total seven points. The full formatted input, reference, and rubric are saved privately.
- Protected boundary: no FOCS 2026 or STOC 2026 content is read. The single selected question becomes development data.
- Generation: 32 responses, temperature one, top-p one, top-k disabled, thinking enabled, total context 98,304.
  Sample seeds start at 20261118. Keep raw special delimiters for extraction without changing sampled token probabilities.
  Save each response immediately. A missing closing thinking delimiter is an incomplete generation, not a zero grade.
- Hardware: one MI350-class GPU, tensor parallelism one, 32 running requests, four-hour maximum allocation.
  Reuse the proposed EAGLE settings. Keep current precision defaults. No training, trainer-head comparison, or weight update occurs.
- Scoring: rubric-item binary satisfaction with the supplied weights. Sum the satisfied weights to a score from zero to seven.
  Show candidate and reference anonymously in both answer orders. Average the candidate scores across orders.
  Preserve raw replies, per-item decisions, usage, parser failures, and both orders.
  Use reference and empty/irrelevant controls. Human calibration remains pending, so grades are provisional.
  Keep provider-specific judgments separate. A Gemini check is a backup or independent audit, never a silent replacement.
- Cost: generation uses at most four GPU-hours and permits at most 2.88 million generated tokens.
  The VM grading budget is $5, with conservative pre-call reservations and usage receipts.
  GPU initialization and container construction count against the allocation walltime.
- Metrics: completed responses, final-answer count, generated lengths, cap-hit rate, reward histogram, score variance,
  all-zero versus uniform-score classification, judge controls and failures, elapsed time, and API usage/cost.
- Baselines and controls: frozen base model only. Reference answer and empty/irrelevant answers test the grading pipeline.
  No learning comparison or significance claim is made.
- Seeds: one reproducible diagnostic group. It is not a three-seed learning study.
- Statistical test: none. This is a single-group feasibility diagnostic.
- Plots: one response-length plot and one provisional grade histogram, with a raw example beside the aggregates.

## Results

Pending submission and execution. Do not infer live job state from this plan.

## Global takeaways

Reserved for the user's interpretation after inspecting the evidence.

## Next steps

1. Submit the committed diagnostic and record the exact Slurm ID immediately.
2. Monitor the allocation, import raw responses, and grade on the VM using the authorized local credential loader.
3. Save raw cases and control grades, upload results, and report limits without claiming judge calibration.

## Questions

Production precision, the broader development split, and a learning horizon remain separate decisions.
