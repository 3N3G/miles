# Inkling 2 Mini: state, scale, and loop readouts

## TLDR

Test the state that actually crosses loop boundaries before fitting a large lens.
The user requested this investigation and explicitly authorized CPU, AMD, and Babel use on 2026-11-21.
This is a bounded exploratory pilot. It does not train or change model weights.

## Background / motivation

Inkling restarts each core pass from its prelude and reads two blocks from the previous pass.
Its message reader normalizes both blocks separately before mixing them.
The code therefore predicts near-invariance to positive changes in either block's scale.
Raw Euclidean convergence can then confuse changes that downstream computation discards with changes it uses.

## Concrete question we are answering

Do positive block rescalings leave later computation unchanged while equal-size directional changes alter it?
If yes, assess loop convergence in normalized bank coordinates and build readouts at this persistent state.
If no, first locate the missing path or numerical effect. Do not interpret a failed control as a new mechanism.

A second exploratory question asks whether loop differences or removal of a prompt-independent component surface unspoken intermediate concepts.
Negative results are useful. They prevent spending a full Jacobian-fitting budget on an unsuitable read site.

## Methodology

- Model: `thinkingmachines/Inkling-2-Mini`, revision `ed40e7da625f594fc9c4990f5f66be24873c22d7`.
- Implementation: Hugging Face Transformers commit `6173a2f679ff4ed393afc499aefc569da7e68bc1`, from `add-inkling2-mini`.
- Numerics: official conversion from released NVFP4/MXFP8 weights to BF16. Record all load warnings. No fresh quantization.
- Compute: at most one AMD MI350 GPU, at most 60 minutes for the first pilot. Keep sequences below 256 tokens.
- Data: published single-token two-hop development prompts from WorkspaceBench, pinned before selection. Up to 12 development items.
- Example smoke prompt from the original Jacobian-lens paper: `The number of legs on the animal that spins webs is`. The unstated bridge is `spider` and the answer is `eight`.
- Four literal completion prompts are engineering controls, not a dataset or benchmark result. Include `The capital of France is`.
- Use the official tokenizer. Print full rendered inputs, token IDs, and top output tokens. Save whether chat formatting or literal completion was used.
- No held-out test set is evaluated. Do not tune and report performance on these development items as a test result.
- Metrics: output KL divergence, maximum logit change, top-token agreement, normalized bank displacement, and unspoken-concept rank/recall@10.
- Baselines: no intervention, same-state replacement, positive rescaling of each bank separately, and direction changes with the same perturbation norm.
- Readouts: active block logit lens, final-reader mixture, normalized bank differences. A difference readout is an unvalidated proposal.
- Must-work control: the final-reader lens reproduces native logits after the same final norm and logit softcap.
- Must-not-work control: shuffled concept labels cannot count as a semantic success.
- Randomness: seeds 0, 1, 2 for directional perturbations. Greedy deterministic readouts need no sampling seeds.
- Statistics: describe the pilot. If reporting an aggregate, show a 95% paired bootstrap interval over prompts. Show individual examples and all failures. No significance claim or model comparison from this pilot.
- Stop early if load conversion fails, native logits disagree, output is nonfinite, or the first forward exceeds the allocation budget.

## Results

Pending. Record actual stages separately: source inspection, weight inspection, model load, forward inference, interventions, and lens fitting.

### Planned plots

1. Bank scale multiplier versus output KL divergence, alongside directional perturbations. Log scale only if needed, clearly labelled.
2. Loop number versus raw and normalized bank displacement. A caption must distinguish these coordinate choices.
3. Intermediate-concept rank across loops, only if the readout gives an informative example. Show the exact prompt and token list.

## Global takeaways

Leave scientific interpretation to the user. Label any proposed interpretation as a draft.
An architectural invariance is not proof that a semantic fixed point exists or that the lens reads reasoning.

## Next steps

1. Verify source and released weight shapes with small CPU checks.
2. Load and validate the pinned model on the authorized GPU.
3. Run scale and direction controls, then inspect readouts.
4. Only after that, propose a full-state, loop-conditioned Jacobian fit and held-out causal benchmark.

## Questions

Challenge values for N, K, script-size limits, and exact permitted read sites remain unpublished as of the inspected page.
