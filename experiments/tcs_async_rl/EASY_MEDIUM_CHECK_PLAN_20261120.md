# Experiment: easy question first, then a conditional medium question

Status: authorized by the user on 2026-11-20.
The user requested one easier question, then medium if its reward distribution is good, to inform the RL run.
This supersedes the unsubmitted proposal's instruction to stop after the first qualifying question.

## TLDR

Run the preselected easy training question first and inspect its complete 32-answer reward distribution.
Review actual mathematical work, both judge orders, paired-reference verdicts, completion and repetition.
If the easy result provides a useful, credible distribution, run the preselected medium question next.
Report both distributions before selecting the RL data. A single question cannot establish corpus-wide feasibility.

## Background / motivation

The original uniform draw selected a hard question without conditioning on difficulty.
Repeated checks on that one question produced constant-zero provisional rewards and repetition failures.
The model and inference backends were being qualified at the same time. This did not measure coverage across TCS.
The user selected this difficulty progression after reviewing those failures and the compute-efficiency concern.
Keep the hard-question runs intact. See `REPAIRED_GROUP_FINDINGS_20261120.md` and `VLLM_GROUP_FINDINGS_20261120.md`.

## Concrete question we are answering

Can the fixed Qwen3.5-4B policy earn credible rubric points on the easy question, with useful reward variation?
If yes, check the next declared difficulty and use both results to inform the bounded RL run.
If no, report whether the issue is missing mathematical work, repetition, saturation or unreliable judgments.
Do not keep drawing new questions until a favorable one appears.
An ambiguous distribution needs review before advancing. Do not silently treat one noisy positive as sufficient evidence.

The first cheap checks are the easy question's supplied reference, empty answer and irrelevant-answer controls.
Run the existing native sampling-support control before generating the group.

## Methodology

- Models: keep Qwen3.5-4B and GPT-6 Luna medium at their pinned revisions/settings.
  `config.sampling_repair_v2.json` owns the model, judge and sampling values.
- Data: use the pinned TCS revision and the two identities in `evidence/alternative_question_proposal_20261120.json`.
  The easy question is an expected rooted-forest cost bound using correlated vertex sampling.
  The medium question transfers a sub-N-log-N Walsh–Hadamard algorithm to a recursive matrix.
  Each was drawn uniformly within its declared difficulty with seed 20261118, before any generated output.
  Exact inputs, references, rubrics and formatted prompts were read and remain in the private proposal directory.
  FOCS 2026 and STOC 2026 remain sealed. No alternative question replaces either preselected identity.
- Configs: `group_check_v14_easy_mi350.json` and conditional `group_check_v15_medium_mi350.json` own run identities and input hashes.
  Unlike the prior failed-seed repair, each new question generates its full fresh group concurrently.
  Keep 32 indexed generation seeds, thinking enabled, 90,000 output tokens and 98,304 total context.
  Keep temperature 1, top-p 0.95, top-k 20 and target-only decoding. Preserve raw thinking and terminal markers.
- Resources: one MI350 GPU per question, at most four hours each, sequentially.
  The two diagnostic allocations have an eight-GPU-hour total ceiling. No optimizer runs during these checks.
  Submit only the easy job initially. A queue wait does not authorize touching another user's or session's jobs.
- Reward: use each question's supplied rubric and the existing final-answer extraction and anonymous two-order judge.
  Preserve every real verdict. Never assign zero on a provider or parser error.
  Missing finals remain ungraded in this diagnostic. Provider credentials remain on the VM.
  Keep the existing $5 per-question judge cap and preserve all private receipts.
- Metrics: show all 32 mean rewards and their histogram, counts at zero and full credit, reward variance,
  response lengths, natural stops, length limits, repetition, order disagreements and all paired-reference scores.
  Review nonzero awarded criteria against raw mathematical work before calling the distribution useful.
  An all-zero group cannot qualify RL. An all-equal group provides no reward-driven GRPO signal.
  A rare-positive or saturated result must be described plainly when deciding whether to advance or change the RL plan.
- Baselines and controls: the frozen policy is the baseline. The supplied reference should earn seven points.
  Empty and irrelevant candidates should earn zero. Inspect all paired-reference verdicts, including anomalies.
  Human calibration and independent-provider validation remain pending, so rewards are provisional.
- Seeds: selection 20261118, generation 20261118 through 20261149, reporting example 20261118.
- Statistical test: none. These are two selected diagnostic questions, not a corpus-level learning comparison.
- Plots: reward points versus answer count, and generated tokens versus answer count, separately for each question.
  Mark the 90,000-token cap. Include the uniformly seed-selected raw answer and its rubric decisions.
- RL handoff: preserve the existing two-update authorization and resources in `RL_SMOKE_PLAN.md`.
  Use the observed easy and medium results to prepare a fresh gate-bound run snapshot.
  No old input hash, reward file or used config may be relabeled for a new question.

## Results

The easy group completed with 15 positive grades, 13 zeros and four capped missing finals.
Both orders agreed on zero versus positive for all 28 graded answers.
The complete setup, mathematical checks and limits are in `EASY_GROUP_FINDINGS_20261120.md`.
This supported the conditional medium submission. Medium completed with 28 zero grades and four missing finals.
See `MEDIUM_GROUP_FINDINGS_20261121.md` for its measured results, example and limits.
The automatic RL handoff stopped. No optimizer update has run.
The easy group still fails the original requirement for 32 nonempty final answers before RL.
Cap handling for that qualification rule is an open question in `QUESTIONS.md`.

## Global takeaways

Reserved for the user's interpretation. The comparison cannot establish representative TCS reward coverage.

## Next steps

1. Validate the easy controls, commit/push the inputs' hashes and configs, and submit its AMD job.
2. Grade, inspect and archive the full easy group.
3. If the distribution is useful and credible, run and inspect medium under the same settings.
4. Report the distributions and update the RL run plan using the measured results.
