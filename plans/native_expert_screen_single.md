# Single-GPU execution of the frozen expert screen

## Scheduling amendment

An AMD development partition has a near-term start estimate. Cancel only this
session's still-pending eight-GPU job541645. It has used zero GPU seconds.
Run one MI210 GPU for at most30 minutes on `devel`. Scientific data, conditions,
controls, seeds, metrics and tests remain exactly as specified in
`native_expert_parallel_screen.md`. The initial frozen MMLU and text manifests
remain unchanged. This preserves the entire prospective text test and the
development derivative search. It uses one worker, rather than eight shards.

## Memory handling

For development derivatives, move saved nonparameter tensors to host memory
during autograd. Keep weight storage on the GPU and do not compute weight
gradients. This changes storage only, not the forward computation or derivative.
Validate a small analytic chain with and without saved-tensor hooks on CPU.
Native forward and exact ablation results complete and save before derivatives.
If derivative execution fails, preserve that failure and these earlier results.
Do not confuse a derivative failure with a failed prospective forward test.

## Allocation

Reserve1800 GPU seconds, count actual allocation including loading. Prior350
seconds remain counted. The program cap is18,000 GPU seconds. Superseded
pending jobs receive their actual zero allocation time in the same ledger.

## Results

Pending.
