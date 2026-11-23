# Find native units that control a block reader

## Question

Does normalization-aware local sensitivity find native units with larger,
pass-dependent effects on Inkling's block reading? The earlier exact replay
showed that the RMS denominator reverses the mean dot-product effect for
layer12 expert45 unit1127. A decoder's direction alone misses that mechanism.

This stage develops a unit-to-reader atlas on the same16 articles already
used for the native routing test. These articles are now development data.
It does not add an independent replication or change the earlier test.
If no stronger units validate, keep the measured single-unit result and
report the limits of first-order ranking. If units validate, freeze their
identities before testing new articles in a separate plan.

## Setup

- Same pinned native BF16 model, grouped expert backend, physical layer12.
- Same16 WikiText-2 validation paragraphs, seed2718,128 tokens each. Example:
  the Calvin McCarty paragraph contains `While K.C.`. Full frozen inputs and
  all token IDs are in `evidence/unit_routing_inputs.json`.
- Outcome: physical layer13 attention reader's mean Block2 weight at token
  positions16..127, on each of the three loop passes.
- Capture a clean full forward and replay only the original expert, short
  convolution, residual sum, BF16 cast and reader. Original expert and block
  states must match exactly before using gradients.
- Make the native grouped activation a differentiable leaf. Compute the
  negative activation-times-gradient score for deleting each native unit.
  Sum its selected-token contributions into the128-by1152 native unit map.
- After the first article's gradients, require a fresh full model forward's
  logits to match the original exactly. No pretrained parameter may have a gradient.
- Also compute a dot-product-only score, holding the block RMS fixed at its
  observed value. This is a diagnostic approximation, not a neural edit.
- Average scores equally over articles. Select four units with opposite mean
  pass1/pass3 signs, ordered by absolute pass3-minus-pass1 contrast. Select
  two units with matching signs, ordered by mean absolute score over passes.
  Break ties by expert and unit ID. Exclude known units45:1127 and3:664.
- Freeze these six identities before exact validation. Validate all six plus
  the two known units by exact native deletion at passes1 and3, on all16
  articles. Existing replay fidelity is an established control. Preserve
  unmodified-model and zero-edit checks from the previous stage.

## Analysis and outputs

Save every unit's scores, fixed selected identities, all exact local effects,
and model/runtime identities. Compare signed prediction error and direction
agreement against zero and dot-product-only predictions. These are descriptive
development results. Do not count unit-pass pairs as independent articles.
No new significance test or primary interval is introduced here.

## Compute

One AMD MI210 allocation, at most10 minutes,600 GPU seconds. Run only after
the third fixed MMLU confirmation order completes. Count all loading time.
The five-hour total cap still applies. Stop at the first failed native replay
or gradient-finiteness guard. Any repair must preserve this question and data.

## Results

Pending.
