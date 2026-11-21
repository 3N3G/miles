# Causal country swaps

## Selection and question

On the 55 development prompts, mean/scale correction recovered more unspoken bridges than the raw logit lens.
In particular, it exposed France in the existing Eiffel-Tower-to-continent task.
This method was selected on development data. No reserved test item has been evaluated.

Does its France-versus-Japan direction represent an argument that the model can use across different downstream questions?
Compare mean/scale correction, the raw logit lens, and the better-predicting affine ridge control.
All three are tested. Success of a readable token list is not assumed.

## Frozen setup

- Same model, tokenizer, BF16 conversion, and code pins.
- Same three fitted calibration seeds. Source maps are active-block readouts.
- Cases: the exact twelve prompts and paired input controls in `evidence/country_swap_cases.json`.
- Primary cases: the three France descriptions that never name France in the input.
- Generalization: explicit capital, language, and continent questions, plus reverse Japan-to-France cases.
- Selectivity: three arithmetic/calendar prompts whose answers should stay unchanged.
- Positive controls: each paired counterfactual input should naturally favor the expected target answer.
- Compute: one MI210 allocation, at most 30 minutes. No model training.

## Intervention

Use the leading-space single-token unembedding rows for France and Japan, multiplied by the final normalization weight.
Pull their score contrast through the chosen affine readout map.
At visits 11, 23, 35, and 41, reflect the active block across that affine contrast's zero plane.
Apply the same concept operation at every token position, as in the published flexible-generalization protocol.
This flips the chosen lens's France-versus-Japan score contrast while preserving its orthogonal component.
For reverse cases the same reflection exchanges the two directions.

Primary dose is one reflection. Doses 0, 0.5, and 2 are recorded as sham/removal/overshoot diagnostics.
Matched-norm controls use random directions with seeds 0, 1, and 2, matching the semantic perturbation norm at each position.
Preserve all attempted outputs, projection values, and perturbation sizes.

## Metrics and interpretation

Report the global top token, the probability mass of complete source/target answer tokens, and their log-probability contrast.
A strong success changes the native top answer to the intended counterfactual while the paired input control also works.
Contrast movement alone is weaker evidence.
Report baseline-unsolved cases separately and retain them in the raw evidence.
The sample is a small mechanistic test, not a population benchmark or significance claim.
Report every fixed case rather than selecting the most successful one.

Stop if the sham changes native logits, if a score reflection is numerically invalid, or if outputs become nonfinite.
