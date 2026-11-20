# Fixed-model rollout check in the compatible existing AMD image

Runtime-control job 534065 tested three cases on the same MI210 allocation.
The primary SGLang image failed basic Torch kernels with host-library injection.
It failed device initialization with its bundled userspace alone.
The existing project image passed FP32 creation and BF16 matrix multiplication.
Its recorded SHA-256 matched `546dafa14c2afdb0a87a89bc892faa1a8238c3f9e25bd860e6715aa1fad2cbcd`.
The passing stack reports Torch 2.12.0+git6bbd260, HIP 7.2.53211, and ROCm 7.2.3 library paths.
This qualifies basic GPU operations. It does not yet qualify Qwen inference, SGLang, Megatron, or RL.

## Exact serving repair

Use the existing image's vLLM 0.29.0+rocm723 for the frozen-model group check.
It contains Transformers 5.16.1 and FlashAttention 2.8.3. It does not contain SGLang or its trainer dependencies.
Run a loopback completion server within the allocation and keep every completed candidate independently.
Use eager target-model decoding without speculative acceleration or prefix caching.
The serving implementation changes. The intended target policy, weights, and sampling controls remain fixed.
Do not claim bit-identical samples or a serving-performance comparison.

Preserve the pinned Qwen3.5-4B weights, exact approved thinking prompt/token IDs, question/input hash,
all 32 indexed candidates and seeds, temperature/top-p/top-k, BF16, 90,000 output and 98,304 context limits,
four concurrent requests, supplied rubric, Luna medium, both anonymous answer orders, and held-out boundary.
Validate the new image's rendered template against the approved string before server startup.
Pass those exact token IDs directly and reject a changed prompt-token count.
Keep incomplete thinking ungraded. Do not insert zero scores for transport or generation failures.

The immutable configuration is `group_check_v6_mi210_vllm.json`.
One MI210, one node, four hours and four GPU-hours remain the bounds.
First verify the image hash and basic Torch operations again. Then load only the already cached pinned model.
Record actual package versions, backend, server argv, raw API replies, every completion, and failures.
The VM performs the existing private Luna grading. No provider key is copied to AMD.

## Relation to the approved RL run

This check reduces uncertainty about frozen-model reward variation and judge transport.
It does not replace the Miles async CISPO implementation or qualify SGLang weight updates.
Keep valid SGLang MI350 job 534034 queued as a separate runtime qualification path.
After a complete group has varied real rewards, submit and debug the authorized bounded Miles RL check.
Any runtime changes for that run require their own preserved snapshot and validation.
All prior failed sources/results remain intact. No new scientific model, question, prompt, reward, or metric is selected.
