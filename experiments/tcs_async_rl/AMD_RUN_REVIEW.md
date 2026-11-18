# AMD submission review

No job has been submitted. The prepared submission is the environment qualification described in
[QUALIFICATION_PLAN.md](QUALIFICATION_PLAN.md). The learning configuration is still a proposal.
This operational record owns the launch procedure. Configuration files own numerical settings.

## Prepared first job

1. Cluster: AMD hpcfund, partition `mi3508x`, account `aviralkumar`, configured allocation QoS.
2. Resource bound: one exclusive eight-GPU node, two hours maximum, at most 16 GPU-hours.
   Early completion or failure releases the node. The image pull and extraction count against this limit.
3. Input: pinned public Qwen weights and synthetic token IDs. No private dataset or judge credential is required.
4. Container: pinned ROCm image digest in `qualification.proposed.json`.
   Extract an unprivileged sandbox to make `/root` dependencies accessible to the cluster user.
   This avoids relying on an unconfigured subordinate-UID range. Actual extraction remains untested.
5. Checks: imports and versions, eight-GPU communication, training CLI imports, model download,
   tiled-head gradient tests, a 98k head benchmark, and single-request long-context inference with EAGLE.
6. Stop on the first failed stage. Reject changed source revisions, login-node execution, silent token truncation,
   benchmark out-of-memory output, and reuse of an existing evidence directory.
7. Scope: no training, checkpoint conversion, async integration, high-concurrency measurement, or judge validation.
   A pass does not establish that the complete RL run works.

## Live checks made during preparation

- Four whole-node MI350 allocations exist. All four were allocated at the latest check.
  The request must queue. Scheduler state is recorded in the linked timestamped evidence, not inferred from this file.
- Apptainer 1.4.5 and Python 3.12.12 exist. Default `python3` is 3.9.21, so the batch file selects Python 3.12.
- No Miles container is installed. The existing vLLM image does not provide this training environment.
- The work filesystem reports about 65 TiB available. This does not establish remaining lab quota.
  The runner requires 120 GiB of free filesystem space before pulling and extracting.
- `~/.config/api-keys.env` exists on the VM. Both provider assignment names were confirmed without inspecting values.
  That file is absent on AMD. Earlier AMD Gemini environment access failed. No credential file was copied.
- Flame is unnecessary for this AMD job. Its changed SSH host key remains unverified.
- The [test-only scheduler check](evidence/amd_submission_check_20261118.json) passed partition access,
  account, allocation balance, one-node size, and the two-hour limit. It returned 256 processors.
  Its provisional start estimate is not a reservation. A follow-up queue check found no qualification job.
  Fourteen CPU checks passed. No container, GPU, or learning check has executed.

## Render and validate, without submission

Run these commands from the synchronized AMD checkout. The job's output directory must exist before submission.
All large files stay under the AMD work filesystem.

```bash
cd /work1/aviralkumar/geney/miles-ian
git pull --rebase
mkdir -p /work1/aviralkumar/geney/tcs-async-rl/runs/qualification-v1
python3.12 -m experiments.tcs_async_rl.amd_preflight \
  --work-dir /work1/aviralkumar/geney/tcs-async-rl \
  --run-dir /work1/aviralkumar/geney/tcs-async-rl/runs/qualification-v1 \
  > /work1/aviralkumar/geney/tcs-async-rl/runs/qualification-v1/job.py
```

Read `job.py`. It contains Slurm directives and a Python command pinned to the exact source commit.
Slurm's `--test-only` validates the scheduler request without submitting a job.
Actual submission uses `sbatch --parsable <job.py>` only after user approval.
Immediately record the returned ID in this workstream and a committed job registry.
Do not alter existing jobs from other tasks.

After completion or failure, upload the run directory with `push-run miles <run-dir>`.
Keep model weights and container files outside the run directory. Record the printed private Hugging Face path.
Sync any registry changes through GitHub. Regenerate the batch file after any source commit changes.

## Intended learning run: reviewed settings and remaining work

The canonical [training proposal](config.proposed.json) contains all current settings.
The [resource budget](RESOURCE_BUDGET.md) explains the arithmetic and its limits.

| Component | Current proposal or decision |
| --- | --- |
| Solver | Qwen3.5-4B, pinned revision, text training plus one prediction layer |
| Data | Ten eligible cohorts, 2,519 questions before development reservation |
| Protected test | FOCS 2026 and STOC 2026, 385 questions, content still untouched |
| Prompt | Dataset problem through Qwen chat template, no reference or rubric in solver input |
| Thinking | Unresolved. Enable thinking is the recommendation for review |
| Length | User selected 90,000 generated tokens. Planned total context is 98,304 |
| Topology | Two trainer GPUs, three two-GPU rollout engines, same node |
| Parallelism | Tensor parallelism two, context and pipeline parallelism one |
| Batch | 64 prompts × 32 responses = 2,048 trajectories and one optimizer update |
| Generation | Temperature one, top-p one, top-k disabled, special tokens skipped |
| Loss | GRPO advantages and CISPO policy loss, recorded rollout probabilities, skip extra actor scoring |
| Clipping | Epsilon offsets 10 and 4. In this implementation the upper ratio weight is five |
| Optimizer | Adam, learning rate 1e-6 constant, decay 0.1, betas 0.9/0.98, zero dropout |
| Memory | BF16, FP32 accumulated gradients, distributed Adam, full layer recomputation, 512-row tiled head |
| Prediction layer | One layer, loss scale 0.2, EAGLE steps/top-k/draft-token settings 2/1/3 |
| Async | Buffer factor one, staleness six, retry unused samples, admission limit 256 |
| Engine limits | 64 active requests each, 0.72 static memory fraction |
| Reward | Supplied weighted seven-point rubric. Luna medium, Gemini Argon backup |
| Provider changes | Explicit, logged, no silent reward mixing. Independent assessment and human calibration remain required |
| Hooks | Generic per-sample judge transport exists. TCS adapter and grading prompt/parser are not implemented |
| Checkpoints | Initial conversion, output path, interval, disk cap, retention, and restart test remain to prepare |
| Horizon and cost | Number of updates, walltime, training seeds, API cap, and total GPU-hour cap remain unapproved |
| Evaluation | Development paper manifest and selection rule remain unapproved. Final test remains sealed |

### What cannot be copied unchanged

Ian's example uses a 2B architecture, private checkpoint/data paths, and a 16-GPU layout.
It requests 16k total context and uses experiment-specific reward/logging/retention hooks.
This proposal changes model dimensions, paths, context, topology, and queue limits.
Per-sample reward calls require `group_rm=false` with the existing judge transport.
Optional logging and retention hooks do not need to block the first qualification job.

The large global batch is a compute choice. It does not require 2,048 resident GPU sequences.
At the worst output cap it contains 184.32 million generated tokens per update.
A whole eligible-data pass would require about 39 such updates before development reservation.
Fully async generation can fetch ahead and retry stale samples. An epoch does not bound judge calls or tokens tightly.
Choose an explicit update count and measured spending limits before launching learning.

### Gates before learning submission

1. Complete the initial environment check, then qualify full-model backward, conversion, async updates, and restart.
2. Approve thinking mode and the paper-level development split. Freeze the formatted solver input.
3. Implement and review the rubric adapter. Calibrate judges against approximately ten human-labeled examples.
4. Establish a permitted credential loader or VM judge service. Confirm authenticated API access without exposing values.
5. Freeze a complete learning plan with horizon, seeds, baselines, metrics, checkpoint policy, and spending caps.

The GPU submission rule comes from the user's global AGENTS instructions: “Ask me first before: ... submitting a new GPU job.”
Preparing files and running CPU checks does not authorize the first allocation.
