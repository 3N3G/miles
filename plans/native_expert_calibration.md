# Native expert calibration and first causal screen

## Question

Can exact removal of layer-12 expert 105 affect numeric behavior, and can the
model answer letter-scored questions with useful clean accuracy? This stage
uses development data only. If task capability is weak, repair formatting on
calibration data before freezing task confirmation. If removal is weak, test
other experts rather than infer that routing means computation.

## Setup fixed before execution

- Model: Inkling-2-Mini, pinned native BF16 conversion, unchanged weights.
- Data: first eight MMLU validation questions per subject for format calibration,
  next sixteen per subject for discovery. Subjects: elementary mathematics and
  high-school biology. Source and converted parquet revisions in the manifest.
- Formats: plain completion and official chat plus an assistant text prefix.
  Choose larger mean gold-letter log probability on calibration only. Report
  both accuracies and total probability mass on A/B/C/D aliases. Exclude no
  discovery item based on the model's answer. Reject length >512 without
  truncation and record it before inference.
- Example mathematics input: Colton and his dad buy a $13 gallon of paint and
  two $9 brushes. Choices $22/$24/$31/$35. Correct letter C before permutation.
- Example biology input: separated fish populations become three species.
  Correct choice: nucleotide sequences of some gene alleles differ.
- Text: sixteen previously observed development paragraphs, 64 tokens each.
  These are explicitly not a fresh confirmation set.
- Edits: expert 105 all visits, each visit alone, and all visits with remaining
  weights renormalized. Experts 45 and 3 all visits. Three frequency-matched
  experts picked from old fit routing, all visits. Whole layer-12 MLP zero is
  a positive control on the first item of each dataset. No weights are fitted.
- Must-not-change controls: identity and impossible expert ID 128, exact logits.
- Seeds: old-fit control selection deterministic with tie by expert ID. No
  stochastic model sampling. Answer-order seeds 0,1,2 are frozen for later
  confirmation, not used to inflate discovery sample size.
- Metrics: correct-letter log probability, four-choice probability and accuracy,
  native token top ten, route weight and fraction selected per expert/pass.
  Text: mean token loss at prediction positions16..62, numeric-target and
  numeric-input strata. Save token losses, full input and top changes.
- Statistics: exploratory screen. No discovery significance claim. Later
  confirmation gets its own primary comparisons and question bootstrap.
- Allocation: one MI210 GPU, at most20 minutes. Reserve1200 seconds before
  submission. The overall budget remains18,000 seconds including prior350.
- Plots wanted after confirmation: expert-removal loss by subject and pass,
  routing enrichment across passes, paired prompt examples.

## Results

The initial allocation was cancelled while pending, using0 GPU seconds. Calibration and forward stages instead executed under native_expert_screen_single.md. No output is attributed to the cancelled allocation.
