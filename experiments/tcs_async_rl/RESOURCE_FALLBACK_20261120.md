# Same-group inference on an available MI210

The MI350 request remains queued. The 2026-11-20 10:10 UTC scheduler check found 11 idle MI210 nodes.
A test-only request predicted immediate allocation. The MI300 alternative predicted a start on the following day.
This fallback stays within the user's approved inference debugging and resource-budget scope.
It answers the same single-group feasibility question. It is not a new learning or hardware-comparison experiment.

## Fixed scientific settings

Use the exact model and input revisions from `GROUP_CHECK_PLAN.md` and its private input hash.
Generate the same 32 indexed candidates with the same per-candidate seeds.
Keep thinking, temperature, top-p, top-k, MTP, and the 90,000/98,304 output/context limits.
Use the same final-answer extraction, supplied rubric, Luna medium, and both anonymous answer orders.
Keep FOCS 2026 and STOC 2026 sealed. No question, prompt, rubric, or score is changed.

Only the GPU class and serving concurrency change: one MI210, four requests at once.
All 32 requests are submitted. The engine queues them under its concurrency limit.
Different hardware and batch scheduling need not produce bit-identical text.
No throughput or quality comparison with MI350 is inferred.

## Memory and time bounds

MI210 provides 64 GiB nominal memory. Runtime must verify the actual capacity and software compatibility.
The pinned model inventory estimates 8.68 GiB of BF16 checkpoint weights.
Four maximum-context trajectories require 13.5 GiB of attention cache, including the selected MTP layer.
Their recurrent state adds about 0.19 GiB. A 20 GiB workspace allowance gives 42.37 GiB total.
The static fraction of 0.72 corresponds to 46.08 GiB, leaving about 3.71 GiB within that allowance.
These are component estimates, not a measured peak-memory guarantee.

Keep a four-hour walltime and four-GPU-hour maximum for this attempt.
Worst-case 90k generation latency on MI210 remains unknown. No silent shortening or truncation is permitted.
The MI350 request remains queued until the fallback demonstrates compatibility or yields a complete usable group.
Do not run duplicate full groups once one has passed the reward gate.

## Runtime repair and validation

Separate AITER, Triton, Torch extension, and Torch compiler caches by image and partition.
This prevents compiled code for the MI210 architecture from contaminating the later MI350 training runtime.
Keep the image and model caches shared and unchanged.
Before loading weights, import the engine and run a small BF16 GPU matrix multiplication.
Then initialize the same model and save every completion immediately.

Preserve submitted v1 and v2 sources/configs. The fallback config is `group_check_v3_mi210.json`.
Run CPU checks and formatting/lint before submission. Record the exact source, job, and limits immediately.
Monitor startup directly. On an unsupported GPU/kernel failure, preserve it and retain the MI350 fallback.
Grade completed outputs on the VM without copying provider credentials.
Upload every terminal run, including failures, and record the measured reward variance honestly.
