# Validate derivative-selected experts before independent task confirmation

## Question

Do the development derivative candidates survive exact native removal and
equal-norm random output changes? This checks a linear approximation before
spending the independent questions. This stage uses development questions only.

## Fixed selection rule

Read the completed screen's32 discovery derivative records. Sum signed routing
weight-times-gradient over each expert's three visits within each question.
Average over questions separately for mathematics and biology. Rank the1536
core expert identities by mathematics mean minus biology mean. Keep the first
two distinct identities whose mathematics mean is positive. Save exact layer,
expert, ranking scores, source hashes and complete condition specifications
before submitting. Expert IDs never transfer between physical layers.

## Data, model and conditions

Use the same pinned native checkpoint,32 discovery questions and original
answer order as the derivative screen. Use its calibration-selected format.
For each candidate remove it on all visits and each visit separately.
Add all-visit renormalization and matched output-norm random edits, seed1729
combined with a stable question hash. Each random edit matches the local
BF16 native deletion norm at the same token/visit and current incoming state.
Later incoming states can differ between conditions. Record actual rounding
error. Include clean, absent-expert and exact router-versus-output-replay checks.
Gold four-choice log probability and accuracy are the outcomes. Derivative
predictions are exploratory, with no independence claim for these questions.

For confirmation, keep a candidate only when its exact mean mathematics loss
increase is positive and exceeds its biology increase. This is a development
selection rule, not a significance test. Expert105 remains a preselected
candidate regardless of this filter. The two held-out cohorts remain unread.

## Native unit observation

After the question edits, observe native SwiGLU activations inside physical
layer12 experts105,45,3,89,41,90 on128 old fit and16 old development paragraphs.
One paragraph has64 tokens. These are the same real WikiText articles as the
pilot, including its previously documented three overlapping development
articles. This is explicitly exploratory. Keep the native forward unchanged.
Audit selected expert order and token positions for every call. The first
observed forward must match a matched-shape unobserved forward exactly.
Store activations, route weights, decoder-column norms, complete texts and
all three visits. No final article states are read for unit selection.

Rank individual units using fitting paragraphs only. Show their held-out
development contexts separately and retain counterexamples. These observations
can guide later exact unit edits but do not identify an arithmetic algorithm.

## Allocation and next stage

One MI210 GPU on AMD devel, at most20 minutes. Reserve1200 GPU seconds before
submission. Actual allocation replaces the reservation afterward. Freeze a
separate confirmation plan and final condition manifest before test inference.
Plots: predicted versus measured deletion effect, candidate/pass task effects,
and native unit examples with activation and route weights.

## Results

Pending.
