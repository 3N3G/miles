# Do phase-sensitive native units control block routing?

## Question and relation to earlier evidence

Expert45 unit1127 at physical layer12 changes sign across passes. Among1318
old fitting token positions routed to this expert on both passes1 and3,
1113 change from negative to positive and2 change in the reverse direction.
The paired mean routing-weighted activities are-0.382 and+0.367. Expert3 unit664
also rises late. These units were selected from earlier decoder alignment and
old native activation data, without independent MMLU test outcomes.

Goodfire already reported loop-dependent outliers and possible block tagging
in this model. This experiment tests specific native unit identities causally.
Their raw output alignments do not by themselves prove a global loop clock.
The intervening convolution and normalization can change a writer's effect.
Therefore measure the model's actual block-mixture weights after native edits.

Does deleting the same unit increase one block's read weight on one pass and
decrease it on another? If so, it supports a pass-dependent routing role.
If effects are small or match random directions, retain phase correlation but
reject the stronger claim that these units are demonstrated routing controls.

## Setup frozen before inference

- Model: the pinned native Inkling-2-Mini BF16 conversion, grouped expert backend.
- Data:16 unused WikiText-2 validation articles, one128-token paragraph each,
  seed2718, in `evidence/unit_routing_inputs.json`. Exclude every previous fit,
  development, final and new text-confirmation article. No topic selection.
- Example: complete first frozen paragraph is printed during preparation and
  saved in the manifest. Use positions16..127 for routing and16..126 for loss.
- Main target: physical layer12, expert45, native hidden unit1127. Zero the
  unit on pass1 alone or pass3 alone at all selected valid token positions.
- Main controls: three equal-output-norm random directions for each pass,
  seeds0,1,2. Compute the counterfactual unit deletion through the unchanged
  native grouped backend, then match its local output norm before convolution.
- Secondary targets: expert3 unit664, each pass1/3; both units jointly on
  each pass1/3; expert105 units156+811 jointly on all visits. Each gets a
  corresponding norm-control seed0. These comparisons are exploratory.
- Include expert45 unit130 on passes1/3 with norm-control seed0. This unit
  was selected by earlier content-decoder alignment and has no comparable
  fitting-phase sign change. It controls for other units in the same expert.
- Identity and absent expert128 unit0 must leave logits exactly unchanged.
  Observe and edit without switching to an eager expert implementation.

## Native measurement and checks

Observe physical layer13's attention-input block reader on all three passes.
Recompute its exact RMS-normalized key, query dot products and softmax weights.
Check that these weights reconstruct the native mixed output with maximum
error<=1e-5 on the first input. The last key at this reader is the immediately
preceding block, Block2. Report its mixture weight, not a trained phase probe.
Verify that a pass3-only edit cannot change earlier-pass reader weights.

For each article and edit, measure the mean change in Block2 mixture weight
at the reader on the same pass. Let C1 and C3 be unit-deletion changes minus
their three-direction random-control means. Bootstrap articles10,000 times,
seed1729. Use Bonferroni intervals across C1, C3 and C3-C1, family coverage95%.
A sign-reversal claim requires opposite signs for C1 and C3 with both corrected
intervals excluding zero. Report the interaction even if this criterion fails.
This tests a local routing effect, not a complete arithmetic mechanism or
proof that the model uses these units as a global iteration counter.

Also save full next-token loss, native-to-edited KL divergence, top-token
agreement, all reader weights, BF16 dose errors and raw examples. Report one
article nearest the median interaction, plus explicitly labeled extremes.
Plot unit activities by pass and causal changes in the native Block2 weight.
During clean forwards, also observe each selected unit's two linear inputs,
the gate and up factors, before their native product. This adds no intervention.
It can show whether the sign change comes from one factor or both. Treat this
as a descriptive decomposition, with the exact softcap constants recorded.

## Allocation and order

One MI210 GPU on AMD devel, at most10 minutes,600 GPU seconds reserved before
submission. Run after the three fixed MMLU seeds, or between them without
inspecting their efficacy or changing their conditions. The global cap remains
18,000 GPU seconds. No new approval is required within the explicit autonomy.

## Results

Completed on all16 frozen articles. Primary deletion-minus-random Block2
weight changes were+0.00008087 on pass1 and-0.00017678 on pass3. Corrected
intervals were[0.00001682,0.00016737] and[-0.00036596,-0.00005282].
The difference was-0.00025765, interval[-0.00051597,-0.00007441]. The
preregistered sign-reversal rule passed. Identity, absent-expert and two
native deletion implementations matched exactly. Full results and limits:
`evidence/native_unit_routing_result.json`. Private immutable artifacts:
`evidence/artifact_archives.json`, stage `native-unit-routing-v1`.
