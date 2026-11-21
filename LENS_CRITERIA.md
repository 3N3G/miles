# What should a useful lens explain?

A useful lens should expose information that the model uses, at a site that actually carries that information.
Readable words and accurate next-token predictions are separate forms of evidence.

For example, consider `The number of legs on the animal that spins webs is`.
The bridge concept is *spider*. The final answer is *eight*.
A next-token predictor can succeed without exposing the bridge.
A useful reasoning lens should expose the bridge and support a meaningful intervention on it.

## What the original methods establish

| Method | What it measures or changes | What to check on Inkling |
|---|---|---|
| [Original logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens), nostalgebraist, 2020 | Applies the output decoder to intermediate activations. The original post explicitly describes a partial view. | Is the activation in an output-compatible coordinate system? Does it include the relevant state? |
| [Tuned lens](https://arxiv.org/html/2303.08112v4), Belrose et al. | Fits an affine translator to the final prediction distribution. The paper also tests causal influence and whether lens and model changes align. | Compare prediction fidelity and causal agreement separately. Do not dismiss the paper's causal tests, or assume they transfer automatically. |
| [Logit prisms](https://neuralblog.github.io/logit-prisms/), Nguyễn, 2024 | Holds nonlinear quantities fixed to decompose output scores into component contributions. | A decomposition on one forward pass does not predict how those quantities change after an intervention. |
| [Patchscopes](https://arxiv.org/html/2401.06102v4), Ghandeharioun et al., 2024 | Uses a model continuation to explain an injected hidden state. | The reader can add computation. Test what came from the source activation versus the target context. |
| [Jacobian lens](https://transformer-circuits.pub/2026/workspace/index.html), Gurnee et al., 2026 | Averages derivatives from intermediate states to final states across contexts and source–target token positions. It probes potential verbalization, including later positions. | Use concept swaps and downstream tasks. A derivative at one point and a corpus-averaged map are different objects. |
| [R-lens](https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers), Blank, Bhatia, and Nanda, 2026 | Changes the backward rules, including normalization and multiplicative gates, to improve readouts. | These are relevance coefficients rather than the forward model's exact derivatives. Validate them against actual interventions. |
| [J++ lens](https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace-lenses), Ayonrinde and Lindsey, 2026 | Weights clustered Jacobians using development tasks, adds relevance propagation, and filters readout tokens. | Keep fitting tasks separate from evaluation. Give all baselines the same vocabulary filter, read count, and top-K budget. |

The [mathematical theory of the Jacobian lens](https://arxiv.org/html/2610.31955v1) explains a preference for directions whose effects remain coherent across contexts and token lags.
Its frequency-weighted extensions alter which temporal scales survive averaging.
Those results motivate a loop-horizon comparison. They do not establish that every coherent direction represents a thought.

[Short Horizons and Sparse Concepts](https://arxiv.org/html/2608.25347v1) separates same-position and broader token effects.
It also finds that these roles do not separate cleanly by simply removing diagonal Jacobians.
Loop depth and future token position should therefore remain distinct axes in an Inkling fit.

## Criteria for this investigation

1. **Read the right state.** Verify the final readout against native logits. Identify paths that bypass the chosen hook.
2. **Expose an unstated variable.** Measure bridge recall and false positives. Count complete concepts, not accidental word fragments.
3. **Predict an intervention.** Replacing a spider representation with a bee representation should favor six legs. Match perturbation sizes and include shams.
4. **Transfer across uses.** The same concept direction should support more than one downstream question and prompt template.
5. **Respect forward invariances.** A readout should not change dramatically under a state transformation that later computation ignores.
6. **Limit decoder inference.** A powerful decoder may solve a problem that its source state had not solved.
7. **Control evaluation opportunities.** Fix N, K, token position, and vocabulary filtering. Selecting the best layer after inspecting test labels inflates recall.
8. **Keep claims reproducible.** Pin model, tokenizer, code, data, and fitting split. Show raw token lists beside aggregates.
9. **Track persistence and horizon.** A readable concept and a lasting edit are different claims. Observe later states after an intervention. A restored original answer can reflect recovery from unchanged input. Test whether a shared decoder transfers across loop passes.
10. **Match actual edits.** Equal coordinate-exchange strength need not give equal perturbation norms. Hold token positions, intervention site, signed coefficients, and edit size fixed when comparing directions.

These criteria are a proposed evaluation rubric, not a claim that the current pilot has satisfied them.

## Existing Inkling results and challenge limits

[Goodfire's release analysis](https://www.goodfire.com/research/interpreting-inkling-2-mini/) already studies information routing, large activation coordinates, and several decoding methods.
It reports strong tuned-lens and coda-lens prediction results, and explicitly cautions that its next-token metrics poorly match the Jacobian lens's objective.
Repeating that comparison alone would add little evidence about reasoning readout.

The [challenge page](https://ndif.us/looped-lens-challenge/) currently asks for ranked tokens at selected layers and one token position.
It does not yet specify numerical N, K, or script-size limits.
A research prototype can inspect multiple state components. Whether all such reads fit the competition contract remains open.
