# A lower-score tie-break would not resolve the four uncertain audits

All four uncertainty flags described the assistant's frozen audit, not uncertainty returned by Luna.
Luna returned one Boolean decision per rubric item, with a short reason.
The saved responses contain neither competing totals nor a confidence field.
They do not expose an internal deliberation transcript.
We therefore cannot backtest the proposed policy from reported alternative scores.

The measured score directions oppose a general downward correction:

| Saved case | Topic | Luna | Frozen assistant audit | Audit confidence |
|---|---|---:|---:|---|
|22|Unary counterexample|4|2|High|
|29|Empty parameter interval|5|7|Medium|
|32|Recursive runtime|3|5|Medium|
|33|Shortest-path witness|1|3|High|
|34|Cube-face construction|5|7|Medium|
|37|Induction base case|0|1|Medium|

These numbers come from the unchanged [disagreement evidence](evidence/judge-comparison-item-v5-20261124-disagreements.json).
Five judge scores are below the audits. One is above.
If the two choices were exactly the observed judge and audit scores, choosing the minimum would fix only case22.
Agreement would become35/40, still below the declared36/40 gate.
This is a hypothetical calculation, not a new judge test.

The two high-confidence disagreements require explicit partial-credit ownership:

- The unary answer proves its objective reaches1 and states the soundness consequence.
  It omits admissible parameter choices and the density calculation.
  Luna knowingly credits the conditional contradiction. The audit requires an established admissible counterexample for that item.
  A repair must specify whether that item depends on the separately scored feasibility and density steps.
- The shortest-path answer supplies a suitable graph and forced opposite endpoint outputs.
  It never rigorously locates nearby opposite outputs for general approximation ratios.
  The audit credits the construction separately. Luna reads the construction item as also requiring that missing transition.
  A repair must distinguish a valid construction from proof of the transition and weighted gap.

High audit confidence does not settle these rubric interpretations.
The existing prompt already discusses component credit and missing prerequisites.
Repeating that guidance is an untested repair, not an established solution.
Keep every original label and result fixed. Resolve the scoring contract before testing another prompt on unused answers.
This review used saved artifacts only: zero paid calls, solver generations or optimizer updates.
