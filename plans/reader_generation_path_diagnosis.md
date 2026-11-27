# Diagnose the failed cached generation control

The first cached diagnostic failed the frozen prefix-equality guard before
its first result. Do not use that generation as an equivalent capability
control. Preserve its log and inputs. No lens scores or states changed.

Use the first frozen arithmetic question to compare32 greedy tokens from:
manual uncached forwarding, generate(use_cache=False), generate(use_cache=True),
and manual native-cache forwarding without an explicit mask. Record every
token sequence and first mismatch before interpreting. Also compare native
prefill logits with an explicit all-ones mask and with a native cache.
The manual uncached reference must match the saved12-token prefix exactly.

Complete the first two frozen questions without cache, maximum256 tokens,
and record raw native thinking/final-channel output. These are diagnostic
examples, not a complete new capability evaluation. This cheaply checks
whether the benchmark's proposed intermediates are written in normal output.

If a cached path reproduces32 reference tokens exactly, it can replace the
failed generation path under the existing completion repair plan. If no
cached path passes, do not run the conditional74-question cached stage.
Keep untested capability states unknown and make no hidden-computation
claim from the bounded token comparison. No prefix guard is relaxed.

One MI210 devel allocation, maximum5 minutes (300 GPU seconds), within the
remaining five-hour program. Preserve the earlier failed diagnostics. This
is a repair diagnosis under standing inference-repair authorization.

## Results

All intended data were written, including two full uncached completions and
the completion manifest. Slurm recordedTIMEOUT during finishing, at321
allocation seconds. Count all321 seconds, including the21-second overrun
beyond the requested300-second limit. Do not relabel the scheduler receipt.

For the first prompt, explicit-one-mask and cached prefill logits matched
the native initial logits exactly. Later cached greedy outputs differed at
index4 for both manual and automatic cache paths. The uncached generation
helper differed from manual forwarding at index22. Neither cached path
passed32-token equality. The larger conditional cached job did not run.
The numerical cause is unproven.

Both full uncached examples answered correctly and wrote the source
intermediate in native thinking text:836 then1575 for(19*44)+739, and64
then128 for((36+28)/3)*6. These are not hidden-intermediate examples.
Only these two received full uncached capability checks. Other gates remain
unknown. Detailed raw text is in evidence/reader_generation_diagnosis_result.json.

Private immutable run archive:
https://huggingface.co/datasets/3N3G/internalization-runs/tree/ca8d3a451dd3326d02dbbf393c0d5b394dfcc819/inkling-interp/reader-generation-path-diagnosis-v1
