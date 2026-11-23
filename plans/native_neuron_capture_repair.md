# Repair native activation observation for grouped expert execution

## Failure and preserved stages

Job541723 completed all32 derivative questions and all32 exact candidate
validation questions. Native logits were unchanged after derivatives, weight
gradients were absent, and native deletion implementations matched. The final
observation stage failed before saving its first paragraph. It assumed the
eager implementation called the activation function once per expert.

The pinned runtime actually dispatches to `grouped_mm_experts_forward` in
`transformers/integrations/moe.py`. It sorts flattened token/expert pairs,
computes all selected branches together, and calls the activation once on
S=sequence_length*8 rows. The observed512 rows were correct for64 tokens.
This failure did not affect derivatives or exact intervention results.

## Exact repair

Preserve the native grouped implementation. Reconstruct its exact
`expert_ids_g,perm=torch.sort(top_k_index.flatten())` mapping. Row i belongs
to expert_ids_g[i], input token perm[i]//8 and weight top_k_weights.flatten()[perm[i]].
Use this mapping only to copy observed activations. Keep the legacy observer
path for an eager implementation. Do not replace the grouped backend.

Validate both paths against analytic teachers. The grouped teacher must have
unequal expert selections and out-of-order top-k slots, with known expert and
token activations. The first native observed forward must match an unobserved
matched-shape forward exactly. Audit three visits per repeated-core router.

Rerun only144 old-text observation forwards,128 fit and16 development. Preserve
all previous results. Native experts105,45,3,89,41,90 remain at physical layer12.
Observe all core routers in the same passes, as already planned. Select any
numeric-routing family using fit paragraphs only, before final task inference.
No held-out task questions or final article activations are read.

## Allocation

One MI210 GPU on AMD devel, at most10 minutes. Reserve600 GPU seconds and
record actual Slurm allocation. Prior failed stage used414 seconds by sacct.
The program cap remains18,000 GPU seconds.

## Results

Completed144 old paragraphs with the unchanged grouped backend and exact logit identity. The128 fitting paragraphs came from106 articles. The16 development paragraphs had3 article overlaps with fitting. These are development data. The numeric family used fitting only. Later article tests excluded these sources. Archive: native-neurons-v2.
