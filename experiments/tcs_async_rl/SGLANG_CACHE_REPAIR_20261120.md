# SGLang master-cache repair

Job 534010 passed both AITER cache imports, then failed before model initialization after 34 seconds.
SGLang's imported DeepGEMM wrapper creates a cache directory even though this BF16 ROCm model does not use DeepGEMM.
Its default path still pointed below the unmounted home directory and raised the same read-only filesystem error.
No Qwen response or reward was produced. This is another cache-configuration failure, not a capability measurement.

The pinned SGLang source provides one authoritative parent, `SGLANG_CACHE_DIR`.
Its lazy `SGLANG_DG_CACHE_DIR` default resolves to the `deep_gemm` child.
Its shared redirect also covers default CuTe, Triton, Inductor, CUDA, TileLang, and FlashInfer cache locations.
Set that parent below the work cache, separated by image and partition.
Keep both independently required AITER overrides and the already explicit Torch/Triton cache paths.
Forward the SGLang setting to Ray workers. Preserve the image and keep the host home unmounted.

The exact pinned path helpers were extracted without SGLang, Torch, or GPU imports.
They selected the work directory, derived the expected DeepGEMM and third-party paths, and wrote a probe successfully.
Evidence and source hash: `evidence/sglang_cache_check_20261120.json`. All 42 CPU checks pass.
Full imports, BF16 GPU operations, model startup, and output remain unverified GPU checks.

Preserve v4 sources/results and cancel own queued job 534013, which has this known import failure.
Use fresh `group_check_v5_mi210.json` and `group_check_v5_mi350.json` snapshots.
Every model, input, sampling, thinking, token-cap, rubric, judge, candidate count, per-hardware concurrency,
allocation, and four-hour bound stays fixed. The MI210 is the active check and the MI350 is its queued fallback.
Inspect startup directly and cancel the unused own fallback after a complete group passes the real reward-variance gate.
