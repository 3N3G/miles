# Independent country-pair transfer and site controls

The France/Japan development check found changes from normalized Jacobian rows.
Test the unused Germany/China row pair, fitted on the same unlabeled text before
these task prompts existed. Preserve the original experiment and all failures.

Model, row fits, fit seeds, interventions, and code pins remain the same.
Freeze fifteen prompts in `evidence/country_transfer_cases.json`: Germany and China,
capital/language/continent relations, explicit names and landmark descriptions,
plus the three original unrelated tasks. Each task has a direct counterfactual
input control. Example: “Fact: The capital of Germany is” versus China. The
landmark version uses the Brandenburg Gate versus the Great Wall.

Primary method: j_unit, all four original visits, dose 2. This dose was selected
on France/Japan development results. Doses 0, 0.5, and 1 remain descriptive controls.
Every dose-2 result must be compared with dose-2 random directions with the exact
same per-position perturbation norm. The original random controls only used dose 1,
so they cannot establish a dose-2 advantage. Record both random doses here.

Three fixed site conditions diagnose where any effect occurs:
- all: visits 11,23,35,41 (primary).
- late: visit 41 alone.
- loop: visits 11,23,35, reusing the visit-35 direction.

Primary metric: alternate next-token answer on cases where original AND direct
counterfactual inputs produce their respective expected next token. Report all
cases as a secondary view. This prevents format failures from counting as swaps.
A raw valid example accompanies each aggregate. Report fit seeds separately and
case-level paired bootstrap 95% intervals. Random-direction seeds are paired
within a case, not independent data. No claim of broad task generality from twelve
related prompts. No source-bank held-out prompts are used.

The exact dose-1 map has a mathematical check: for unit country rows a,b,
H=I-2(a-b)(a-b)^T/||a-b||² swaps their two projections and preserves vector norm.
Dose 2 overshoots that exchange. It is not norm-preserving. Report this distinction.

All shams must equal native logits. Inspect the complete formatted prompts and
native outputs. Do not tune prompts or doses after model outputs are seen.
Budget: three sequential MI210 jobs, each at most 15 minutes. Archive each run.
