# Experiment: one sparse operator across Inkling's three passes

## TLDR

Status: approved for execution. The user selected this direction with
“Follow your suspicion” and approved the proposed allocation with “yes” on
2026-11-22. The preapproval config is preserved in `plans/archive/`.

Approved limit: one MI210 job, one GPU, at most 30 allocation minutes.
The released model's weights remain frozen. Only small surrogate models train.

## Background / motivation

Weight tying gives a direct reason to share a model of a sublayer's operation.
It does not guarantee shared activation distributions or shared output readouts.
The earlier captures contain whole-layer outputs, not the exact sublayer pairs
needed for this test. Reusing those captures as transcoder training pairs would
test a different function with incomplete inputs.

The target is physical layer index 11, the final layer of the first core block.
This site was fixed before collecting new sublayer data. It also has earlier
causal observations, so a later comparison will not require a new site search.

## Concrete question we are answering

Can one sparse surrogate approximate the same local sublayer in all three
passes with accuracy close to three separate surrogates?

If yes, inspect which learned features persist or change their use, and test
their native-model effects by visit. This supports a reusable local description,
not a universal lens or a complete recurrent algorithm.

If no, compare a dense surrogate and a ridge baseline. A weak dense ceiling
indicates inadequate data, capacity or optimization. It does not disprove sparse
operator explanations. A strong dense ceiling but weak sparse fit identifies
the chosen sparsity/capacity setting as the failed approximation.

The cheap failure test is held-out sublayer prediction. Exact native-output
replay must work before any model-intervention result is trusted.

## Methodology

- Model: the pinned Mini checkpoint and Transformers commit in
  `evidence/source_pins.json`, using the same official BF16 conversion as earlier.
- Operator: `language_model.layers[11].mlp`. Its input is after block mixing,
  cross-loop message gating, and normalization. Its output is before the short
  token convolution. The router and experts are inside this pointwise operator.
- Visits: the same physical module's calls 0, 1 and 2, labeled passes 1, 2 and 3.
  Do not equate a physical module with one execution.
- Fit data: the existing 128 WikiText-2 train paragraphs from
  `evidence/transport_calibration.json`.
- Validation data: the existing following 16 paragraphs. They have been used in
  earlier research and are explicitly development data.
- Final data: 32 randomly selected eligible paragraphs, one per article, from
  WikiText-2's validation split, excluding exact token-prefix matches to prior
  calibration paragraphs.
  Fix their identities before model capture. Their new sublayer outputs remain
  unused until every surrogate checkpoint and ridge setting is frozen.
- Rendering: plain text, no added special tokens, first 64 model tokens.
  Keep token positions 16 through 63 for fitting and local evaluation.
- Concrete fit example: a paragraph beginning “Further , Trappist monk , poet
  and spiritual writer Thomas Merton used Kilmer 's poem as a model”. The saved
  token IDs and decoded 64-token input, rather than the untruncated paragraph,
  define the actual model input. Print one full rendered input per split.
- Token count: 6,144 fit and 768 validation rows per pass. Final data supply
  1,536 rows per pass. The same token positions are aligned across passes.
- Seeds: 0, 1 and 2. Retain each result, rather than treating seeds as independent
  data replicates or pooling their token rows into a larger sample size.

### Models and fair comparisons

1. Shared TopK transcoder: affine encoder, ReLU, retain at most 64 activations,
   linear decoder. Width 4,096, input and output dimension 2,304. No pass label.
2. Three pass-specific TopK transcoders, each with the same width and sparsity.
   Also evaluate each on the other passes to produce a transfer matrix.
3. Shared dense ReLU surrogate with width 4,096, as a capacity/control ceiling.
4. Shared ridge regression. Choose regularization from 0.01, 0.1 and 1 using
   validation prediction error only.
5. Shared and pass-specific training-output means. Shuffled input-output pairs
   are a must-not-work control for the implementation's small synthetic test.

Train for 20 epochs using Adam, learning rate 0.001, batch size 256, no weight
decay. Normalize inputs and outputs with training-only vector means and scalar
root-mean-square scales. Normalize sparse decoder columns after each update.
Choose each checkpoint by validation mean squared error, never final data.

Each training row appears once per epoch. The shared model sees all passes.
The three specific models each see one pass. Thus both conditions see the same
total rows and epochs, while the specific condition uses three times the
parameters. Report both parameter count and update count. This is not a claim
of equal per-model training data or parameter count.

### Metrics, controls and statistics

- Primary: prediction squared error divided by final-data target variance,
  computed separately per pass. Compute document-level error arrays.
- Primary contrast: shared minus matching pass-specific normalized error,
  averaged across the three seeds and passes for each paragraph. Report a
  paired 95% bootstrap interval over the 32 final articles, 10,000 resamples, seed
  1729. Show all per-pass and per-seed values. No binary superiority claim.
- Secondary: centered output cosine, active-feature counts, feature firing
  rates by pass, and cross-pass transfer of the separately fitted models.
- Routing diagnostic: expert frequency and aligned-token expert-set overlap
  by pass. It measures reuse, not expert semantics or causal specialization.
- Must work: same-input native operator replay agrees exactly, an observational
  capture hook leaves model logits unchanged, and synthetic planted sparse
  operators can be learned on separate synthetic test inputs.
- Must not work: a shuffled-pair sparse surrogate fails to match the planted
  teacher as well as a correctly paired model. A mean surrogate must not be
  called an interpretable operator even if later model computation repairs it.
- Stop on nonfinite values, hook-count mismatches, split contamination or native
  replay failure. Preserve failures. Do not alter the frozen final split.

### Native-model checks and feature interventions

After local prediction, evaluate the first eight fixed validation paragraphs.
Do not use final paragraphs to choose interventions or semantic labels.

Compare native behavior with the shared surrogate at all three visits,
matching specific surrogates, and pass-specific mean replacement. Use seed 0
for native-model diagnostics. Report token-distribution Kullback–Leibler
divergence, next-token agreement and target-token negative log likelihood.
These are fidelity measures, not evidence of semantic understanding.

Select four active shared features by summed last-token activation on those
eight validation paragraphs. Save the selection rule and IDs before edits.
For each feature, paragraph and pass, subtract `activation * decoder_direction`
from the native sublayer output at the last token only. Keep the native
approximation error. This isolates the feature edit from surrogate replacement.
Compare a matched-norm random output direction using random seed 1729 + feature
ID. Zero-activation edits remain in the report as shams, not silently excluded.

Save the largest output-logit changes for both directions and the feature's top
training contexts per pass. Textual feature labels are hypotheses written after
inspection. Raw examples and counterexamples must accompany any such label.

The feature selection and intervention stage are exploratory. They cannot
establish general semantic validity, a unique feature basis or a full loop rule.

## Results

Not run. CPU validation covers implementation controls only.

### Exact plots wanted

1. Local normalized error by pass, same method color throughout, paired
   paragraph bootstrap intervals. Caption states the measured sharing cost.
2. The three-by-three transfer matrix, training pass on rows and evaluated pass
   on columns. Color encodes normalized squared error, without changing scales.
3. Feature firing rates across passes, with one raw training context beside each
   highlighted feature. Label selection as exploratory.
4. Native replacement fidelity by condition. Show one complete input and its
   native and replacement token outputs beside the aggregate.

Bar-chart axes start at zero. Report seed variation separately from article
uncertainty. Do not treat three seed points as a trend.

## Global takeaways

Leave measured results and uncertainties here after execution. Any interpretation
is a draft for the user. No mentor-facing or public report is authorized.

## Next steps

Prepare and validate code, freeze the final paragraph identities, then obtain
the required 30-minute compute-budget decision. If approved, submit the single
job through `src/submit_stage.py`, record it immediately, monitor it, and upload
the run using `push-run inkling-interp`.

## Questions

The direction and proposed 30-minute budget are approved. Record execution and results without requesting the same permission again.
