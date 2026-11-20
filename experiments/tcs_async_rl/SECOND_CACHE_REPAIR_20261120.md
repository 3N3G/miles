# Second AITER cache repair

Job 533964 failed after 83 seconds, before model initialization or generation.
It produced zero Qwen responses. There are no measured Qwen rewards from this attempt.
The original AITER JIT cache repair worked: `module_aiter_core` compiled in 55.5 seconds under the work cache.

## Failure and scope

A later import followed `aiter.ops.attention` into `csrc.cpp_itfs.utils`.
That file independently chooses `AITER_ROOT_DIR`, defaulting to the unmounted home directory's `.aiter` subdirectory.
Its `BUILD_DIR` creation raised `OSError: [Errno 30] Read-only file system: '/home1'`.
The separate `AITER_JIT_DIR` setting does not control this path.

This identifies another container cache-path error. It does not establish a GPU compatibility or model-capability failure.
The queued MI350 job 533702 uses the same affected image without this second override.
Cancel that own queued job and replace it with a committed repaired snapshot, preserving its original source and config.
The MI210 failed run and its logs are archived privately under `miles/group-check-v3-mi210`.

## Exact repair

Set `AITER_ROOT_DIR` to a separate writable C++ cache below the bound work filesystem.
Keep image/partition cache namespaces, the original explicit JIT cache, and the unmounted home directory.
Forward this additional cache setting to Ray workers for the later RL run.
Do not change HOME, mutate the image, expose credentials, or alter any scientific setting.

New submitted snapshots will be `group_check_v4_mi210.json` and `group_check_v4_mi350.json`.
They retain their predecessor's candidate count, seeds, question, model, thinking, token limits, rubric, judge,
GPU allocation, concurrency, and four-hour bound. Only cache configuration and run identity change.
The MI210 is the immediate compatibility/generation attempt. The MI350 request is its queued fallback.
Cancel the unused own fallback after one complete group passes the real nonconstant-reward gate.

## Validation before submission

Extracted the exact pinned `AITER_ROOT_DIR` and `BUILD_DIR` assignments without importing AITER or Torch.
With the explicit override, they selected the work directory and successfully wrote a probe file there.
The source hash and narrow test scope are in `evidence/aiter_cpp_cache_check_20261120.json`.
All 42 CPU checks pass, including both explicit cache variables and partition isolation.
Actual full import, BF16 GPU kernels, model loading, and generation remain GPU-job checks.

Inspect startup directly on each repair. Preserve every terminal result and upload it with the job registry updated.
Do not translate missing generation or failed judgments into zero rewards.
