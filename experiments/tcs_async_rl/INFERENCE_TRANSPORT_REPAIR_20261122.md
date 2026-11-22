# Inference repair: exclude solver thinking from reward requests

The approved continuation failed before its first new optimizer update. A completed HTTP request contained
4,671,119 bytes of whitespace-heavy solver thinking. The VM server rejected its body at the two-MiB limit.
Its generic exception handler returned HTTP 502, which correctly stopped training instead of inventing a reward.
The scheduler reports job 538647 failed after 996 seconds. Both trainer ranks had restored iteration 64,
scheduler counter 2,080 and learning rate 1e-6. No new checkpoint or training point exists.

The exact saved sample is index 2171, capped at 90,000 generated tokens without a final answer.
Its compressed artifact SHA-256 is `34638833f18756e2251880cc667ba79caeff4d728806b9473be1a7688455e589`.
A local HTTP replay reproduced status 502 and `HTTPRequestEntityTooLarge` before any scorer or provider call.
This diagnosis does not establish why the model generated the long whitespace loop.

## Exact repair and validation

- Protocol 3 sends only the already extracted final answer and the recorded model-limit flag.
  Raw thinking, tokens, policy replay inputs and finish status remain in the original generated artifact.
  The scorer still accepts delimited legacy diagnostic requests. It rejects requests containing both representations.
- The scorer passes the same answer bytes to the unchanged Luna-medium rubric prompt and anonymous order swaps.
  A capped empty final still requires its real paired zero judgments. Provider failures still raise.
- Increase the HTTP body allowance to 32 MiB for long final answers. Preserve HTTP errors such as 413.
  Log exception classes, status codes and stack locations without headers, exception text or credentials.
- The exact failed sample now sends 133 JSON bytes and returns HTTP 200 with its cached paired 0/0 verdicts.
  The replay made zero provider calls and left the shared judge ledger unchanged.
- All 103 TCS CPU checks passed using unittest discovery with aiohttp. Black, isort and Ruff passed.
  The checks include final-answer transport equivalence, the large-thinking cap, real HTTP handling and failure evidence.

## Replacement run, within the original limits

The user's standing inference-repair authorization covers this replacement. Configuration: `rl_easy_overfit_v15.json`.
Keep Qwen3.5-4B thinking on, Luna medium, prompt v3, all 20 easy questions, seven-point rubrics,
solver prompt, seed, sampling, optimizer, two actor GPUs and six rollout GPUs. FOCS/STOC 2026 remain sealed.
Resume the unchanged iteration-64 checkpoint. Target 40 new updates, global steps 65–104.

The replacement receives at most **3h43m** on eight MI350 GPUs. Together with the failed 16m36s allocation,
the ceiling is 3h59m36s, or 31.9467 GPU-hours. Disable automatic Slurm requeue on the replacement.
The same $5 ledger inherits all 552 settled calls, $0.6374188 estimated spend and zero pending reservations.
This allowance includes all earlier judge qualification and proof-audit calls. Do not reset it.

Preserve the failed run, old frozen source and configurations. Keep foreground supervision through termination,
checkpoint inspection, immutable result upload and native W&B verification. Review entropy, divergence,
gradients, weight publication, cap/length behavior, judge disagreements and actual proofs at least hourly.
Use within-continuation matched-question visits. Never treat the grader change as learning.

## Evidence and operational paths

- [Failed runtime archive](https://huggingface.co/datasets/3N3G/internalization-runs/tree/81df47c1f98e5b11e4058793116ca06a6a462842/miles/easy-overfit-v14).
- Exact failed generation: immutable trace revision `bfe8913b20ab3a0e14c63f19780076422bbbf0fd`.
- VM validation: `~/.cache/miles/tcs-async-rl/runs/easy-overfit-v15/exact-failed-answer-transport-validation.json`.
- VM replacement directory: `~/.cache/miles/tcs-async-rl/runs/easy-overfit-v15`.
- AMD replacement directory: `/work1/aviralkumar/geney/tcs-async-rl/runs/easy-overfit-v15`.
- Source checkpoint: `/work1/aviralkumar/geney/tcs-async-rl/runs/easy-overfit-v12/checkpoints`, iteration 64.
- Session `01a1ed24-1032-7731-910b-824f484f5f71`, home VM, original directory `/home/ubuntu/miles`.
- Workstream `tcs-async-rl-recon`. No duplicate job or subagent is authorized by this repair.
