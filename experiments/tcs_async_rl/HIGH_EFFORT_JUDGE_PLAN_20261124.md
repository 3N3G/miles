# Experiment: higher-effort judging, followed by bounded prompt optimization

## TLDR

The user approved Luna high, followed by more prompt optimization if it does not work.
This supersedes the previous medium-effort choice and one-repair stopping rule.
Keep the existing $2 comparison ledger within the $25 all-stages cap.
The already approved easy/medium screen and twelve-hour RL plan remain unchanged.

## Background / motivation

The fresh forty-answer packet gave paired-medium 29/40 exact score matches and reference-medium 32/40.
All controls passed, but neither met the grading gate.
Some disagreements concern equivalent mathematical arguments and independent partial credit.
See [the prior comparison](JUDGE_COMPARISON_RESULTS_20261124.md).

## Concrete question we are answering

Does higher reasoning effort, with the prompt fixed, reproduce the frozen mathematical audits more reliably?
If yes, use that fixed recipe for the approved GPU stages after controls pass.
If no, inspect the actual disagreements and optimize the prompt within the same task and budget.
Assistant agreement remains a diagnostic proxy for mathematical accuracy.

## Methodology

- Model: the same GPT-6 Luna API model, changing medium to high reasoning effort only.
  Preserve the existing 8,192-token initial verdict budget and bounded unfinished-verdict expansion to 32,768.
- Data: the unchanged fresh forty-answer packet, SHA256 `888fc0f8f27146e1e25addc0862396ca499c2dd286e89e33dc2e798fe6770a30`.
  Its labels remain SHA256 `63aba9a7b191f2386b57bdb758af12b17691768c223bb2c3f77a759f4cf46dba`.
  It contains 32 easy answers and eight from the original medium question, on seventeen distinct questions.
  One example proves an interval empty by comparing its endpoints to sqrt(2n).
- Prompt: unchanged `rubric_judge_reference_v2.txt`, original weighted binary rubric, final answer only.
- Baselines: reuse the exact same-packet paired-medium results and the reference-medium results.
  Verify packet, label and baseline-prompt fingerprints before reuse. Do not charge cached calls again.
- Controls: supplied reference must score seven, empty and irrelevant answers zero, on all seventeen questions.
  Forty candidate calls and 51 controls require 91 base requests. Preserve and charge all retries.
- Metrics: exact weighted-score agreement, case-balanced item agreement, absolute score error and false full-credit awards.
  Keep the observed 90% exact / 95% item gate. Review substantive false awards separately.
  Keep question-cluster bootstrap intervals descriptive. No significance or population-accuracy claim is planned.
- Seed: retain the packet's 20261124 selection seed. No provider seed or temperature is added.
- Spending: use the same append-only owner, `judgments-v1/budget.jsonl`, which has $0.4516942 settled already.
  The comparison cap remains $2, leaving $1.5483058. Never reset the ledger or extend a cap silently.

If high effort fails, use the existing two packets only as prompt-development data.
Record each exact prompt and its reason before calling it.
Try at most three further general prompt candidates in this bounded round.
Prioritize mathematical equivalence, component ownership, then short contrasting examples of valid and invalid reasoning.
Keep model, effort, original rubric items, weights, answer extraction and controls fixed.
Do not insert answer-specific expected scores or change audit labels to fit outputs.
Separate the two documented first-packet audit corrections from its original frozen labels.

Select the development recipe by the existing exact-agreement/error/cost rule.
Then freeze a third disjoint forty-answer packet and its assistant labels before validation calls.
Exclude both earlier packets and previously inspected sample identities.
Retain 32 easy answers and eight unused medium answers, without new solver generations or held-out reads.
Report reused question identities and the lack of independent human calibration.
Do not repeatedly inspect and retune against that final validation packet.
If no recipe passes or the comparison cap prevents validation, preserve the evidence and report the remaining limitation.

## Results

Completed: high effort matches 36/40 total scores and 96.79% of items.
All 51 controls pass. The observed gate passes, so further prompt optimization is unnecessary in this round.
The four remaining disagreements and immutable evidence are in
[the standalone results](HIGH_EFFORT_JUDGE_RESULTS_20261124.md).
No GPU stage has yet run.

### Planned figures

- Exact-score and item agreement against the frozen baseline, with descriptive question-cluster intervals.
- Per-answer score errors, with a raw mathematical example beside each aggregate.
- Actual paid usage across fixed-effort and prompt-development stages.

## Global takeaways

Reserved for the user. A prompt-development winner is not an independent mathematical accuracy measurement.

## Next steps

1. Run the approved fixed-prompt high-effort test and controls.
2. If needed, optimize on development answers and use one untouched disjoint validation packet.
3. Continue the previously approved GPU stages only after the grading gate passes.

## Questions

The user's latest instruction authorizes both high effort and further prompt optimization.
No new approval is needed for these steps within the stated bounds.
