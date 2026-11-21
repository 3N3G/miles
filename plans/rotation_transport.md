# Rotation and transport lenses

## Authorization and scope

The user explicitly requested continued creative experimentation on 2026-11-21.
They suggested applying the latest rotation for the remaining depth.
Earlier CPU, AMD, and Babel authorization remains in effect.
The first execution is capped at one GPU and 60 minutes. No model weights change.
Repairs and bounded continuations use the same model, questions, calibration source, and criteria.
New scientific conclusions and human-facing reports remain review drafts.

## Question

Does continuing the latest representational change expose useful internal variables before the final decoder?
Two vectors do not determine a unique rotation. Test the minimum two-plane rotation and a population-fitted rotation separately.
If either predicts future states on separate text and improves meaningful concept readouts, test causal semantic swaps.
If a rotation fails, distinguish inaccurate dynamics from a readout that merely predicts the final answer.
Use stronger linear transport as a control before rejecting transport-based lenses.

## Fixed setup

- Model and implementation: the exact revisions in `evidence/source_pins.json`.
- Calibration: 128 unlabeled WikiText-2 train rows, each truncated to 64 tokens. Sample selection is frozen in `evidence/transport_calibration.json`.
- Calibration validation: the following 16 disjoint rows. Use these only for geometric regularization or damping decisions.
- Fit seeds: 0, 1, 2, each selecting 96 of the 128 fit rows.
- Collect token positions 16 through 63. This gives 4608 fitting token rows per seed without relying on the first few context positions.
- Main concept development data: the twelve previously observed WorkspaceBench items plus 43 new development items. The split is frozen before model capture.
- Reserve the remaining 43 source-bank items. Only their identities are recorded. Do not inspect their model outputs while iterating.
- Inspect all original inputs and raw outputs. Retain wrong answers and stale source labels rather than silently correcting them.

## Read sites and target

Capture the active block and the model's final-reader mixture after each of 48 visits.
The final reader must reproduce the model's native output at visit 47.
The target for geometric prediction is the final reader before final normalization.
Also retain the two saved banks at each loop boundary, derivable from active-block endpoints.
Source states at a given visit may use only that visit and earlier visits on the same prompt.

Fixed semantic read budget: visits 11, 23, 35, and 41, final prompt token, ten ranked tokens per visit.
Show both full-vocabulary and identical letter/digit-filtered results for every method.
Any other-layer exploration is labelled exploratory and cannot replace the fixed-budget score.

## Candidate methods and controls

1. Raw active-block and mixed-state logit lenses.
2. Mean and scale correction only, using unlabeled calibration statistics.
3. Minimum-plane angular extrapolation from the latest pair of normalized states, repeated for the remaining visits.
4. The latest population-fitted orthogonal rotation, raised to the remaining-step power.
5. A directly fitted orthogonal map to the final-reader coordinates.
6. A ridge-regularized affine state predictor as a stronger control.

For step 3, test damping factors 0, 0.25, and 1. Select a factor only by separate-text geometric error.
For the affine ridge fit, select among scales 0.01, 0.1, and 1 by separate-text prediction error.
Orthogonal fits use a fixed identity-favoring regularizer of 0.1 after global variance normalization.
Do not choose either using bridge labels.

## Evaluation

- Primary diagnostic: future-state centered squared error and cosine on calibration-validation text.
- Secondary diagnostic: final-token agreement and output KL, on separately observed development prompts.
- Interpretability: complete single-token bridge recovery at the fixed read budget, false-positive inspection, and behavior-correct/unspoken subsets reported separately.
- Must-work controls: final-reader equality, fitting a known synthetic rotation, and correct horizon-zero behavior.
- Must-not-work controls: shuffled state pairs and prompt-independent mean readouts.
- Statistics: paired bootstrap 95% intervals over documents or prompts, not over token rows. Three fit seeds are shown separately and averaged only explicitly.
- No claim of improvement from the best layer or the best method selected on development labels.

## Causal validation if a candidate is useful

Use a concept direction from the fitted map with the model's own unembedding.
Swap it with a same-type concept at the source activation, preserving the orthogonal component.
Compare correct downstream answer changes with matched-norm shuffled or random directions.
Prefer the published flexible-generalization country tasks and an unspoken country description.
Freeze exact prompts, alternatives, intervention sizes, and controls before this stage runs.
Report failed interventions as well as successful ones. Do not call readout recovery a causal result.

## Plots and records

Plot geometric prediction error by visit, fixed-budget bridge recovery by method, and any validated causal-swap result.
Each aggregate must have a raw input/readout/output example.
Save model/code/data pins, fit settings, all learned artifacts, and scheduler receipts.
Upload run results privately with `push-run`.
