# Compare the current native block reader with the partial block buffer

## Question and interpretation

Does decoding a current layer's own depth-attention read improve intermediate
token recovery compared with decoding its partial block buffer? The native
reader selects and normalizes several blocks. Pure positive rescaling of one
buffer cannot change the token order through this bias-free output head, so
the test instead changes the direction read by the lens. No weights are fitted.

If the reader improves recovery, it supports this native read site as a useful
starting point for a token lens. If it fails, the verified local unit-to-reader
predictor remains useful for routing analysis but has not become a better
semantic lens. This is a test of an architectural consequence, not a claim
that the previously selected eight units directly encode these concepts.

The user authorized autonomous expert research up to five GPU hours and
explicitly requested ramifications for general lens finding. This is within
that question and remaining budget. No competition submission is made.

## Model and fixed lenses

Use the same pinned production Inkling 2 Mini, BF16 conversion, grouped
experts and reference convolution. No cache, one prompt per forward. Read
only visits12,24,34,35,36,43,44,45,46,47, fixed to the public starter's choice.
Read the last prompt token at every visit. Use zero-based effective visits.

The baseline applies the final RMS norm and bias-free unembedding to the
state entering the decoder layer, as the public starter does. The alternative
applies the identical head to output zero of that layer's native
attention_input module, before cross-loop gating. This is the query-path
depth-reader activation where multiple native read streams exist. Each lens
uses only its one current native activation, with no cross-visit combinations.

Return each method's ten tokens per visit after dropping special/control
tokens and duplicate case/spacing forms. Do not filter or rerank using prompt
text, gold labels, token IDs from the prompt, or other readouts. Use the same
cleanup for both methods. Softcap is unnecessary for ranking because it is
monotone. No trained weights, prompt-aware lexicon, external model or judge.

## Data, examples and controls

Use three source families from the pinned WorkspaceBench repository commit
abd152e20aa2fa5d7df7f01ade6759ebe0f858fb. Sample32 questions per family with
seed161803. These families have not been used in prior inference here.

- Arithmetic intermediates: e.g. compute (19*44)+739, intermediate836,
  final1575. Score the first source-defined intermediate.
- Sequential arithmetic: e.g. double6 then double again, intermediate12,
  final24. Score the first source-defined intermediate.
- Direct readout: e.g. the day before Saturday, answer/intermediateFriday.
  This measures answer visibility before generation, not hidden reasoning.

Require the label to have a single-token vocabulary form. Exclude prompts
containing the label as a whole word/number and previously used prompts.
Use a fixed effort-zero chat wrapper matching the official practice format.
Freeze exact token IDs, source files/hashes and complete examples in
evidence/native_reader_lens_inputs.json before any inference. Add all ten
public practice questions unchanged as a separate descriptive set.

On the first question, verify all read hooks leave native logits exactly
unchanged. Reconstruct the native final head exactly. Require48 layer visits
and exactly ten chosen read sites per method. Trace only the initial prompt
forward. Then greedily generate up to12 tokens without read hooks, disabling
cache, to inspect capability and whether the intermediate was spoken.
Generation and capability gates do not construct or filter lens outputs.

## Metrics fixed before inference

Primary endpoint: whether any of a method's ten readouts contains the exact
source-defined intermediate, after case/spacing normalization. Use the64
arithmetic and chain questions. Report matched hit counts, both kinds of
discordance, and the paired recovery-rate difference. Bootstrap questions
10,000 times with seed1729 for one95% interval. No visit choice, label choice
or method tuning follows evaluation. Results generalize only to this sample.
Source items are authored benchmark tasks, not observed human reasoning.

Report the32 direct questions and ten practice questions separately. Report
all visits' hit counts descriptively. Report capability and unspoken-label
subsets as secondary checks. Exact matching can miss valid synonyms and
numeric paraphrases. Include the first alternative-only success and the
first baseline-only success in frozen input order. If no discordance exists,
show the first question. Print full model input, output and both readouts.

## Plots

Bar chart of exact recovery by method for arithmetic, chain, direct and
practice. Axes start at zero. Label each bar with hit count and denominator.
Separate line chart for each method's per-visit arithmetic recovery, with
zero-based effective visit on x and hits/64 on y. No smoothed curve.

## Compute and archive

One MI210 devel job, maximum20 minutes (1200 GPU seconds). Submit after the
phase job finishes to keep one allocation at a time. Reserve before submission
and count the whole allocation in the five-hour program. Preserve this plan
before adding Results. Archive frozen inputs, outputs, native controls and log.

## Results

Completed the106-question fixed readout comparison. The primary64 arithmetic
questions gave1/64 exact intermediate recoveries for each method, with no
paired discordances. The empirical paired bootstrap collapses to[0,0], which
does not establish population equivalence. Direct source-label recovery was
18/32 for the buffer and14/32 for the current reader. Public practice was6/10
versus4/10. All106 initial generations reached the12-token budget unfinished.
Those capability and unspoken gates remain unknown in this bounded test.

A source audit found six direct questions with subjective preference labels
from the source model. Their aggregate is not factual answer accuracy.
Frozen-label counts are preserved. Source subgroups and raw examples are in
evidence/native_reader_lens_data_audit.json and evidence/native_reader_lens_result.json.

The longer completion shortcut failed the planned native prefix guard. A
bounded diagnosis completed the first two questions without cache. Both gave
the correct final answer but wrote the intermediate in native thinking text.
No larger cached stage ran. The remaining62 arithmetic gates are unknown.
This test does not establish a hidden-intermediate benchmark for Inkling.

Private immutable initial readout archive:
https://huggingface.co/datasets/3N3G/internalization-runs/tree/7a872cf5e31507a581c2daaa2dd74ee7035774ff/inkling-interp/native-reader-lens-v2
