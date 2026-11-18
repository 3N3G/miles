# Experiment: TCS async RL with Ian's Miles fork

Status: reconnaissance complete. Experiment design is a draft, not an approved run plan.
No GPU jobs or paid judge calls have been submitted.

## TLDR

The code is staged on AMD. The dataset has 2,519 eligible questions and 385 held-out questions.
The main untested pieces are the ROCm training stack, long sequences, and a calibrated reward adapter.
There is now a concrete [configuration proposal](config.proposed.json), with resource calculations in
[RESOURCE_BUDGET.md](RESOURCE_BUDGET.md). Missing paths and decisions are explicit null fields.
See [open decisions](../../QUESTIONS.md) before implementing experiment choices or requesting resources.
The next proposed allocation is a bounded [AMD environment qualification](QUALIFICATION_PLAN.md).
Its [operational review](AMD_RUN_REVIEW.md) separates the prepared first job from the unfinished learning configuration.
The user created [3N3G/miles](https://github.com/3N3G/miles), forked from Ian's repository.
VM Git writes use a dedicated SSH deploy key. Read/write access is verified and preparation commits are pushed.
The AMD checkout reads the user fork over HTTPS and is synchronized.

## Background / motivation

The user requests fully async RL on `ian-and-friends/tcs_v3` using Ian's Miles fork.
Ian supplied a Qwen3.5-2B CISPO command from a different experiment.
He suggested Qwen3.5-4B and `gpt-6-luna` at medium effort for this work.
The user selected a 90,000-token solver output cap. Solver thinking mode remains open.
The 90k option comes from Ian's suggestion. It is not an estimate of required response length.

The original command's main settings are `--fully-async`, GRPO advantages, CISPO,
rollout log probabilities, no actor scoring forward, epsilon offsets 10/4, and learning rate 1e-6.
It uses Adam betas 0.9/0.98, weight decay 0.1, temperature 1, top-p 1, and top-k -1.
Its async settings are staleness 6, unused-sample retry, buffer factor 1, and 2,048 concurrent samples.
It also trains one MTP layer with loss weight 0.2 and uses EAGLE speculative decoding.
These are reference settings. They are not an approved TCS configuration.
The user explicitly confirmed that this command illustrates how to call Miles.
Replace experiment-specific settings when preparing the reviewed TCS configuration.

## Concrete question we are answering

Proposed: does async CISPO improve Qwen3.5-4B's rubric coverage over its frozen base model?
Compare models at the same solver budget on papers excluded from training.
A positive result would justify a larger controlled study. A null result would require diagnosis.
Interpretation belongs to the user. No result or interpretation is available.

The cheapest uncertain steps come first: reward calibration, runtime compatibility, then long-sequence feasibility.
Check whether base-model groups have varied rewards before allocating a learning run.
Groups with identical rewards supply no GRPO learning signal.

## Methodology

### Models and source revisions

- Miles: `80a407b98a819add3e17383dc3238e3df9296f98` from Ian's fork.
  The user fork adds preparation records. Record its exact commit for each submitted job.
- Candidate solver: `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
  Only its config and tokenizer were downloaded. No model weights were downloaded.
- Candidate judge: `gpt-6-luna`, medium reasoning effort. API access has not been tested.
  Official docs confirm Chat Completions support. Sampling knobs such as temperature must be omitted with reasoning enabled.
- The 4B model has 32 layers, hidden size 2,560, 16 attention heads, four KV heads, and a 262,144-token context.
  It has one MTP layer and tied embeddings. The supplied 2B dimensions and checkpoints cannot be reused.
- Keep pipeline parallelism at one in the proposed first check. Ian's changelog describes a tied-MTP/PP>1 gradient issue.

### Data and protected boundary

Source revision: `ian-and-friends/tcs_v3@8e486f2d980d926fa5543efe7f1f2adf475ce3ea`.
All ten eligible Parquet files were downloaded with an explicit allowlist.
FOCS 2026 and STOC 2026 question content was not downloaded or inspected.

| Cohorts | Questions | Cohort/paper IDs | Role |
| --- | ---: | ---: | --- |
| FOCS 2023/2024/2025 | 753 | 190 | Eligible for training/development |
| STOC 2023/2024/2025 | 725 | 184 | Eligible for training/development |
| SODA 2023/2024/2025/2026 | 1,041 | 262 | Eligible for training/development |
| FOCS 2026 | 201 | Not inspected | Final held-out evaluation |
| STOC 2026 | 184 | Not inspected | Final held-out evaluation |

Eligible data contains 1,235 proof-writing and 1,284 proof-strategy questions.
Difficulty labels are 2,283 hard, 216 medium, and 20 easy.
Every problem, answer, and rubric is nonempty. Every rubric starts with `Total: 7 points`.
Rubrics are prose, not JSON. No normalized exact problem or answer duplicate exists within eligible data.
Cross-boundary duplicates and near duplicates are not yet audited. Do not claim they are absent.
The 2026 held-out cohorts use a different question-generator split from eligible data.
Held out means excluded from this experiment's training and tuning. It does not establish exclusion from pretraining.

One example was selected with seed 20261118: SODA_2023, paper
`Online_Sorting_and_Translational_Packing_of_Convex_Polygons`, node `node_25`.
It asks for constructive polygon-packing approximation guarantees and has six weighted rubric items totaling seven points.
The complete problem, reference answer, rubric, and both tokenizer template previews remain in the private local audit.
No dataset text is copied into this plan or proposed for a public GitHub push.

Qwen tokenizer counts, before the chat template:

| Field | Median | 95th percentile | Maximum |
| --- | ---: | ---: | ---: |
| Problem | 574 | 1,032 | 2,026 |
| Reference answer | 1,589 | 4,058 | 7,466 |

Proposed: reserve development papers within eligible cohorts using a frozen paper-level manifest.
Do not choose checkpoints, prompts, rewards, or hyperparameters on the 385 held-out questions.
The fraction, seed, duplicate policy, and exact manifest require review. No split has been created.

### Solver and reward interface

Proposed solver input: the dataset's problem through the model's chat template.
No system prompt, extra paper context, filtering, or thinking mode has been selected.
Keep reference answers and rubrics on the judge side. Preserve cohort, paper, and question identity.

The application must implement the reward adapter. Ian's `adaptive_rl.miles_hooks` is not in this repository.
The existing generic transport supplies HTTP, retries, and concurrency, not a TCS grading policy.
The reference's `--group-rm` calls its reward hook with a list.
`miles.rollout.llm_judge.reward` accepts one sample. A direct substitution would violate that contract.
Standard eval rollout also rejects `--group-rm` in this source.
Review whether to keep a batched wrapper or use per-sample reward calls.

Exact hook calls in the pinned source:

| Hook from Ian's command | Call site | Arguments |
| --- | --- | --- |
| `adaptive_rl.miles_hooks.reward` | [`rm_hub/__init__.py`](../../miles/rollout/rm_hub/__init__.py#L103) | `await hook(args, samples, **kwargs)` in group mode |
| `adaptive_rl.miles_hooks.log_rollout` | [`rollout/metrics.py`](../../miles/ray/rollout/metrics.py#L78) | `hook(rollout_id, args, samples, rollout_extra_metrics, rollout_time)` |
| `adaptive_rl.miles_hooks.keep_two_checkpoints` | [`megatron_utils/actor.py`](../../miles/backends/megatron_utils/actor.py#L867) | `hook(args, rollout_id, checkpoint_dir, hf_checkpoint_dir)` after saving, on rank zero |

The first call is entered by `sglang_rollout.py:414` after a group's generation completes.
The logging and post-save hooks are optional. A truthy custom logging return skips default rollout logging.
The existing per-sample generic judge entrypoint is `miles/rollout/llm_judge.py:171`.
Request Ian's referenced module and its imported helpers, judge prompt, parser, and non-secret configuration.

The user selected the supplied seven-point rubrics with Luna at medium effort and a Gemini backup.
No new rubric is needed. The adapter must preserve rubric items/weights and parse the judge's response.
Retain per-item decisions and the supplied weights. Log truncated answers, missing answers, and parser failures explicitly.
Do not convert judge transport failures into zero rewards. Do not invent a scalar 1–10 grading scale.
Calibrate on about ten human-labeled eligible examples. Swap candidate/reference order and average.
Use an independent second-provider assessment before reporting correctness or learning gains.
Earlier TCS judge checks disagreed on both dataset-reference controls. Agreement alone is not accuracy.

### Resource proposal and checks

AMD source: `/work1/aviralkumar/geney/miles-ian`. Its clean checkout follows `3N3G/miles/main`.
It contains Ian's pinned runtime baseline plus preparation records. No Miles runtime code was changed.
Apptainer 1.4.5 is available. The only existing image found was a vLLM ROCm inference image.
A candidate Miles image exists at:
`docker.io/rocm/sgl-dev@sha256:25597c01ab3fc8e77bfeb48434262c535080dc6857c8d591165b4b9304425019`.
Registry metadata reports Linux/amd64 and 19,387,976,373 compressed bytes.
It has not been pulled, converted, or tested. Its packaged dependencies may differ from Ian's fork.
The dated tag `miles-rocm10-mi35x-20261016` does not exist. Do not rely on the docs' example tag pattern.
Image construction and GPU tests must run through Slurm, not on the login node.

The reference requests eight actor GPUs plus eight rollout GPUs: sixteen GPUs in total.
The resource-based proposal is one eight-GPU AMD node, two actor GPUs and six rollout GPUs.
Candidate actor layout is TP2/CP1/PP1. Candidate rollout layout is three TP2 engines, 64 running requests each.
The [configuration proposal](config.proposed.json) retains Ian's 64x32/global-2048 batch and optimizer settings.
It lowers async admission to 256 and makes FP32 gradient accumulation explicit.
Model-state memory does not require lowering global batch size. Accumulation separates it from resident GPU sequences.
The compute sensitivity table compares that batch with a 256-sample alternative. No smaller batch is silently selected.
These are untested choices requiring approval. Preserve existing jobs and use the current scheduler limits.
Observed mi3508x MaxTime is four days. This differs from the older global note's twelve-hour limit.
The first proposed technical allocation is one node for two hours, at most 16 GPU-hours.
It makes no judge calls and does not train. The learning allocation and duration remain unselected.

Selected solver output cap: 90,000 tokens. Planned matching total context: 98,304 tokens.
This leaves 8,304 tokens for the full prompt, including its chat template and any approved instructions.
Set the trainer sequence configuration and engine context capacity consistently with 98,304 tokens.
The reference's 16,384-token context flags cannot serve this output budget.
The 90k cap is a ceiling. Generation can stop earlier. Measure actual lengths and cap-hit rates.
Reference answers have median 1,589 and 95th-percentile 4,058 tokens. They do not measure Qwen's generated thinking.
Context means prompt tokens plus generated tokens. Qwen's generated thinking tokens count toward its output budget.
For arithmetic only: a 2,000-token prompt plus 128,000 generated tokens needs 130,000 total tokens.
A 131,072-token context would fit that example. A 128,000-token total context permits only 126,000 generated tokens.
The phrase "128k" must therefore be resolved to an exact token count and either output or total context.
`--rollout-max-response-len` limits generation. `--rollout-max-context-len` limits prompt plus generation.
The request builder enforces the smaller of the output cap and remaining context (`generate_endpoint_utils.py:67`).
`--sglang-context-length` sets engine capacity. `--seq-length` sets trainer sequence configuration.
This source derives `max_position_embeddings` from `seq_length` in `megatron_utils/arguments.py:35`.
The judge has a separate input/output budget. The solver's output becomes part of the judge's input.
Dynamic batch token budgets do not split one oversized sample. Tiled logits do not remove attention memory costs.
Ian's new tiled-head tests are registered for CUDA. ROCm correctness and MTP need an explicit check.
Qwen GDN context parallelism requires the FLA backend in this source.
Validate kernel imports, model conversion, forward/backward, EAGLE, in-place pause, weight updates, and save/reload.

The reference's 64x32 batch contains 2,048 trajectories.
It permits 184,320,000 output tokens at 90,000 each, or 262,144,000 at 128,000 each.
One pass over all eligible prompts with 32 samples permits 7.25B or 10.32B output tokens respectively.
These are upper bounds, not expected runtime or token use.
The earlier 4x8 pilot suggestion was not selected. Use the canonical JSON for the current proposal.
Pilot duration, proposed topology/concurrency, seeds, and spending limits remain unapproved.

### Metrics, controls, seeds, and statistical test

Proposed primary metric: independent, calibrated, weighted rubric coverage on development papers.
Report fully satisfied rubrics separately. Coverage is not the probability that a proof is correct.
Controls: frozen base model, empty/irrelevant negative answers, and independently checked positive answers.
Record raw examples, lengths, token-cap rates, group reward variance, judge failures, costs, and policy staleness.
Proposed full comparison uses three training seeds. A technical smoke test cannot establish a learning gain.
Proposed single primary test: paired permutation test over per-paper scores, averaging training seeds first.
Report 95% intervals and specify their resampling unit. Any extra comparisons need a correction decided in advance.
Review the exact metric, seeds, evaluation sample, and test before running.

## Results

No learning or GPU-runtime results exist for this experiment.
The new CPU estimator passes four tests. Public tensor headers establish 4.206B text parameters,
0.121B MTP parameters, and 0.334B vision parameters. See the linked resource report for assumptions.
Judge access checks found no OpenAI credential in VM/AMD process environments.
The AMD process-environment Gemini credential was rejected as `API_KEY_INVALID`.
The user then identified `~/.config/api-keys.env`. An authorized names-only check found both provider assignments on the VM.
Values were not parsed, extracted, displayed, or copied. No API request used this file's credentials.
Flame's login VM is RUNNING and reachable through IAP, but its changed SSH host key still needs verification.
No generation request was sent. Provider validity remains untested with the newly identified credential file.
The dataset audit, tokenizer previews, and registry checks are reconnaissance evidence only.
Private local evidence: `/home/ubuntu/.cache/miles/tcs-v3-recon/`.
The pinned dataset subdirectory contains `audit.json`, shard hashes, and `random_nonheldout_example.json`.
The audit used PyArrow 26.0.1 and Transformers 5.23.0 without PyTorch or GPU inference.

Proposed plots: development rubric coverage versus updates, response length versus updates,
token-cap rate versus updates, and staleness versus updates. Use separate plots and show seed uncertainty.
Keep the final held-out comparison sealed until the training and selection protocol is frozen.

## Global takeaways

Pending results and the user's interpretation.

## Next steps (most to least important)

1. Keep the located credentials private. Authenticated tests with those credentials remain pending after the names-only check.
2. Review the resource-based configuration, resolve thinking mode/limits, and freeze the data manifest and adapter.
3. Review the prepared qualification job, then validate the remaining full training path before a learning allocation.
4. Upload every experiment run through the repository's result workflow and record its exact source revision.

## Questions

See [QUESTIONS.md](../../QUESTIONS.md). No research choice in this draft is silently implemented.

## Sources

- [Pinned dataset](https://huggingface.co/datasets/ian-and-friends/tcs_v3/tree/8e486f2d980d926fa5543efe7f1f2adf475ce3ea)
- [Pinned model configuration](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json)
- [Luna model limits and pricing](https://developers.openai.com/api/docs/models/gpt-6-luna)
- [Reasoning request compatibility](https://developers.openai.com/api/docs/guides/deployment-checklist)
- [Ian's validation and limitations](../../changelogs/2026-10-24-fused-lm-head-and-mtp.md)
- Previous TCS evidence: `/home/ubuntu/tcs/notes/exp4_judge_focused_review_20261020.md`
