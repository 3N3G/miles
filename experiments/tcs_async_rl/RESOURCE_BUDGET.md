# Qwen3.5-4B async RL resource budget

The arithmetic supports testing **one eight-GPU MI350 node**, with **two trainer GPUs and six rollout GPUs**.
This is a configuration proposal. There is no measured memory peak or throughput result yet.
The canonical settings are in [config.proposed.json](config.proposed.json).
The [calculator](resource_estimate.py) generates [these estimates](evidence/resource_estimate.json).

## Available hardware

The [scheduler snapshot](evidence/gpu_snapshot_20261118.json) was taken at 2026-11-18 19:51 UTC.

| Partition | Nominal topology | Nodes in pool | Idle nodes at snapshot |
| --- | --- | ---: | ---: |
| mi3508x | Eight MI350-class GPUs per node | 4 | 0 |
| mi3501x | One MI350-class GPU per scheduler node | 8 | 2 |
| mi3008x | Eight MI300-class GPUs per node | 2 | 0 |
| mi2508x | Eight logical MI250 GPUs per node | 10 | 0 |

These are shared pools, not an exclusive allocation. A full eight-GPU node must queue at this snapshot.
No existing job was stopped, modified, or resubmitted. Flame's repaired login VM is RUNNING and reachable through IAP.
SSH is blocked by a changed host key. Its current GPU inventory is therefore not yet known.
The mi3508x partition reports a four-day walltime limit. That is not a proposed run duration.

AMD specifies 288 GB HBM and 8 TB/s bandwidth for MI350X/MI355X.
The calculator conservatively treats 288 GB as 288e9 bytes, or 268.2 GiB.
It uses the MI350X dense BF16 peak of 2.3096 PFLOP/s per GPU for compute sensitivity.
The exact device model, usable HBM, and achieved rate still need a job-local probe.

## Parameters and trainer memory

Pinned public checkpoint tensor headers give these counts, without downloading model weights:

| Component | Parameters |
| --- | ---: |
| Text model, including tied embedding/output weight | 4,205,751,296 |
| One multi-token prediction (MTP) layer | 120,599,552 |
| Vision model | 333,514,240 |

The full header inventory is [saved here](evidence/model_tensor_inventory.json).
Text plus MTP training therefore covers about 4.326B parameters. Vision is excluded from this text-only trainer estimate.
The rollout estimate conservatively includes all checkpoint weights, about 8.68 GiB in BF16.

With BF16 parameters, FP32 accumulated gradients, and distributed FP32 Adam, model-state bytes are
`P * (6 + 12 / (DP * CP)) / TP` per rank. TP is tensor parallelism, CP is context parallelism,
and DP is data parallelism. This accounts for model weights, gradients, master weights, and Adam states.
The optimizer's exact shard groups and tensor padding must be confirmed in the selected runtime.

At 98,304 total tokens, with full activation recomputation and sequence parallelism:

| Trainer layout | Model/gradient/Adam state per GPU | Saved hidden checkpoints per GPU |
| --- | ---: | ---: |
| One GPU | 72.5 GiB | 15.5 GiB |
| Two GPUs, TP2/CP1 | 36.3 GiB | 7.7 GiB |
| Four GPUs, TP2/CP2 | 24.2 GiB | 3.9 GiB |

Saved hidden checkpoints count one layer input per recomputed layer. They are not the total activation peak.
Temporary activations, kernel workspaces, communication, weight publication, and allocator fragmentation remain unmeasured.
The proposed two-GPU layout has enough modeled margin to justify profiling it before reserving four training GPUs.

Without tiling, one BF16 vocabulary-logit tensor would be 22.7 GiB per rank at TP2/CP1.
Ian's 512-row tiled LM head reduces that individual buffer to about 121 MiB per rank.
MTP uses the tiled path too. This does not remove full-attention compute or all intermediate tensors.

## Rollout memory and concurrency

The text model has eight full-attention layers, four KV heads, and 256 dimensions per head.
BF16 key/value memory is `8 layers * 2 K/V * 4 heads * 256 dimensions * 2 bytes = 32,768 bytes/token`.
One full MTP attention layer increases this planning count to 36,864 bytes/token.
At 98,304 context tokens, that is **3.375 GiB per trajectory across an engine's TP ranks**.
Linear-attention recurrent and convolution state adds about 49.1 MiB per trajectory in the simple state model.

For each proposed two-GPU rollout engine:

| Component | Planning size |
| --- | ---: |
| Static-memory budget at fraction 0.72 | 386.2 GiB |
| 64 full-context KV caches, including one MTP layer | 216.0 GiB |
| 64 linear-attention states | 3.1 GiB |
| All checkpoint weights | 8.7 GiB |
| Explicit assumed workspace reserve | 20.0 GiB |
| Remaining planning margin | 138.5 GiB |

The reserve is an assumption. Prefix-state pools, CUDA/HIP graphs, speculative trees, and runtime buffers need measurement.
This is not proof that SGLang's peak allocation will equal the table.
Three such engines give **192 active generation slots**. The proposed async admission cap is 256.
Admitted samples can queue or wait for grading. They do not all hold maximum-size GPU KV caches together.
The original 128-running-requests setting would exceed this modeled static budget if all requests reached full context with MTP.

## Which batch settings to copy

Yes: retain Ian's `rollout_batch_size=64`, `n_samples_per_prompt=32`, and `global_batch_size=2048` as a reference proposal.
Gradient accumulation processes those samples in microbatches. Global batch 2,048 does not require 2,048 resident sequences.
Retain his learning rate 1e-6, Adam betas 0.9/0.98, weight decay 0.1, clipping offsets 10/4,
rollout log probabilities, MTP weight 0.2, and async staleness limit 6 as proposed starting values.

The proposal changes hardware layout, context lengths, running-request limit, and admission concurrency.
It also makes FP32 gradient accumulation explicit, matching the memory formula above.
It retains MTP/EAGLE for compatibility testing. Their speed benefit on this exact AMD setup is unknown.
No topology, batch, or optimizer proposal in this file has been submitted as a training run.

Compute, rather than model-state memory, may make 2,048 samples per update unsuitable for the first pilot.
The table below assumes a fixed 4,000 aggregate generated tokens/s across all rollout engines.
That rate is a sensitivity point, not a measurement or prediction. Judge latency and retries are excluded.

| Generated tokens per sample | Output tokens per 2,048-sample update | Rollout time at the assumed rate |
| --- | ---: | ---: |
| 4,096 | 8.39M | 35 minutes |
| 16,384 | 33.55M | 2.33 hours |
| 90,000 | 184.32M | 12.8 hours |

The compute model includes dense projections, causal full attention, backward, and one full recomputation.
It includes the extra tied-head projection for MTP. GDN recurrence, communication, and pipeline bubbles are excluded.
At 90k output plus a 2k prompt it estimates about 6.15 PFLOP per training sample.
For two trainer GPUs, a 2,048-sample update takes 2.52–7.57 hours at assumed 30%–10% of MI350X peak.
These are conditional calculations. They are not walltime guarantees.
Async rollout and training overlap, so stage times should not simply be added.

An eight-prompt by 32-response pilot would use 256 samples per update, eight times less work at the same lengths.
That is a review option, not a memory requirement or a silently selected replacement.
Measure actual generated lengths and stage throughput before choosing the learning horizon and final batch.

## Rubric grading and API access

Use the supplied seven-point rubrics with GPT-6 Luna at medium effort, as the user requested.
All 2,519 eligible rubrics have parseable weighted items totaling seven points.
They contain three to seven scored items. No new rubric is needed.
The adapter must format the request, parse per-item decisions, preserve weights, and report failures explicitly.
The user also requested a Gemini backup. Provider/model identity must remain attached to each grade.
Do not treat a provider change as the same reward distribution without checking agreement.

[Initial access checks](evidence/judge_access_20261118.json) used existing process environments only.
No key value was returned to the conversation or copied.
OpenAI's credential is absent from both VM and AMD process environments, so Luna access is not tested.
The AMD process-environment Gemini credential returned HTTP 400 with `API_KEY_INVALID` on a metadata GET.
The user then identified `~/.config/api-keys.env` and requested a check of key existence, not values.
That [names-only check](evidence/credential_presence_20261118.json) found `OPENAI_API_KEY` and `GEMINI_API_KEY` on the VM.
Values were not parsed, extracted, displayed, or copied. No authenticated request used those file credentials.
The AMD invalid-key result does not establish anything about the file credentials.
Those initial probes did not validate either provider. A subsequent authorized VM-only credential loader
validated Luna metadata access and completed six grading control calls. Gemini returned HTTP 401.
The reference scored 7/7 and the empty/irrelevant controls scored 0/7, in both orders. Human calibration remains pending.
No credential was transferred to AMD. The online RL judge route is not deployed.
The one-group diagnostic has a $5 grading cap and stores usage receipts. Its initial controls cost an estimated $0.014775.
For scale only: at standard Luna input pricing, 184.32M judge input tokens cost $18.432 before reference/rubric text,
judge output, swapped presentation, or retries. Actual judge tokenization and input length require measurement.

## Validation and next step

Four CPU tests check hand-derived KV geometry, optimizer sharding, batch/concurrency separation, and token/time arithmetic.
Formatting and lint pass. No GPU test, model conversion, or long-context measurement was run.
The JSON file is a configuration proposal, not a native Miles config loader or a runnable launch command.
Required paths, the TCS reward adapter, and run/spending limits remain explicit null fields.
The user approved solver thinking. The canonical config now enables it and keeps Luna at medium effort.
The next GPU action should be a bounded memory/throughput qualification of the proposed topology, after review.
The prepared [first qualification job](QUALIFICATION_PLAN.md) covers the environment, output head, and one inference engine.
It is bounded to 16 GPU-hours. Full-model backward and simultaneous training/generation need a later integration check.

## Sources

- [Pinned Qwen configuration](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json)
- [AMD MI350X specifications](https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html)
- [Megatron distributed optimizer memory model](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html)
- [Megatron memory-estimator assumptions, including CP state sharding](https://docs.nvidia.com/nemo/megatron-bridge/nightly/training/memory-estimator.html)
- [Luna limits and standard pricing](https://developers.openai.com/api/docs/models/gpt-6-luna)
- [Gemini model catalog](https://aistudio.google.com/docs/models)
