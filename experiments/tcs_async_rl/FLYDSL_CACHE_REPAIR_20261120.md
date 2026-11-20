# MI350 model warmup and FlyDSL cache repair

Job 534034 ran on an AMD Instinct MI350X with 309,220,868,096 bytes of GPU memory.
The primary image passed its SGLang import and BF16 matrix control in 59 seconds.
This separates the MI350 result from the earlier MI210 base-kernel incompatibility.

The model warmup then called AITER's FlyDSL GEMM compiler. The compiler tried to create
a lock under `/sgl-workspace/aiter/aiter/jit/flydsl_cache`, inside the read-only image.
It raised `OSError: Read-only file system`. The job failed after 179 seconds, without a completion or grade.
The existing AITER Python/C++ and SGLang cache overrides had passed their import paths.

## Owner and exact repair

The pinned AITER `__init__.py`, lines 94–97, selects its bundled FlyDSL cache
only when `FLYDSL_RUNTIME_CACHE_DIR` is absent. Set that supported override before import.
The FlyDSL PA decode tuner also documents `FLYDSL_AUTOTUNE_CACHE_DIR` for its writable results.
Set both directories under the existing image/partition cache namespace on `/work1`.
Forward both variables through the existing Ray runtime environment for the future RL run.
Do not modify either container or change which mathematical kernels are requested.
Empty writable caches may require compilation. No warmup-time claim is made.

The CPU preflight check verifies both paths. Actual compilation and model startup still need the replacement allocation.
The ready native argument checks remain valid because no model or optimizer argument changed.
The actual MI350 memory measurement exceeds the conservative decimal-capacity assumption in the earlier budget.

## Replacement plan

Use fresh `group_check_v8_mi350.json`, a pinned checkout and run directory.
It retains all scientific fields from v5: one question, exact input, 32 indexed candidates,
thinking, sampling, 90k output / 98,304 context caps, BF16 and 32-request admission.
The saved engine arguments confirm EAGLE with steps/top-k/draft-token settings 2/1/3.
An earlier version of this note incorrectly described native decoding as target-only.
The vLLM diagnostic uses target-only decoding. The native settings have not changed.
The single-MI350/four-hour resource bounds remain. Preserve v5 and its uploaded failure evidence.
Require actual model warmup and generation. The already-running vLLM group remains independent.
All grades remain provisional. Do not manufacture nonconstant rewards by altering the rubric or question.

This is an authorized inference cache repair. No new scientific condition or provider is selected.

Private immutable failure evidence: https://huggingface.co/datasets/3N3G/internalization-runs/tree/3fea142be65322087281db759783f8890f161d51/miles/group-check-v5-mi350

## Second cache failure and replacement

Job 534296 passed the FlyDSL failure point, then failed after 249 seconds.
The speculative-tree kernel calls SGLang's separate `kernels/jit/utils/compile/loader.py`.
Its build lock creates the directory returned by `compile/cache.py:317–319`.
That owner reads `SGLANG_JIT_CACHE_DIR`, then defaults to `~/.cache/sglang/jit`.
The older `SGLANG_CACHE_DIR` override does not own this compiler's cache.
The default tried to create `/home1` in the read-only image. No completion was generated.

Set `SGLANG_JIT_CACHE_DIR` under the existing image/partition work cache before import.
Forward it to Ray workers. Preserve the EAGLE settings and every scientific input.
Use fresh `group_check_v9_mi350.json`, source checkout and run directory.
The same one-MI350/four-hour limit applies. Preserve and upload the v8 failure.
CPU checks verify the explicit cache path and partition separation.
Actual model warmup and generation remain required.
