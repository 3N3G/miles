# Native repetition failure and bounded-sampling repair

Job 534333 completed all 32 requests in 28 minutes 50 seconds on one MI350X.
Thirty-one answers ended naturally and contained final answers.
Sample 25 used all 90,000 generated tokens without closing its thinking block.
Its tail repeats `Wait.` thousands of times. The scheduler reported a length limit, not an engine crash.
No request was retracted. Preserve the complete raw response and all earlier answers.
This rules out the assumption that the current sampling settings reliably finish this group.
It does not establish a model-wide truncation rate or mathematical inability.

## Hypothesis and exact repair

Unfiltered sampling can enter an unproductive trajectory before a repetition loop.
Test bounded top-p 0.95 and top-k 20, retaining temperature 1 and the 90k output / 98,304 context caps.
The pinned [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/README.md)
recommends those values for general thinking tasks. This is a testable repair hypothesis, not a known cure.
The card also recommends a presence penalty. Do not add it here.
Miles cannot replay history penalties during training. Its sampling-support contract rejects them.

Bounded sampling requires support-normalized behavior and actor log probabilities.
Miles enables this path automatically for positive top-k or top-p below one.
The current fused LM-head backend rejects it. Select the existing Torch backend in a fresh configuration.
Do not add unsupported penalties, ignore the validation, or train against unfiltered denominators.
The same six rubric items, primary judge, effort, controls and final-answer extraction remain unchanged.
No new question or held-out content is selected.

## Prospective validation and run order

Use `config.sampling_repair_v1.json` and `group_check_v10_mi350.json` from a pinned checkout.
First request 16 technical-control tokens through the native SGLang request schema.
Require complete support metadata, each sampled token in its support, at most 20 support tokens,
and finite support log probabilities whose exponentials sum to one within 0.0001.
Save the full control output before validation. This control receives no rubric score.
The Engine convenience wrapper omits these request fields, so call its tokenizer-manager owner with `GenerateReqInput`.

Next generate the previously failed sample 25 with its original seed and full 90k cap.
Require a natural stop and nonempty final answer. Preserve its output and stop if validation fails.
Only after that check, generate the other 31 original indexed seeds concurrently.
Sample 25 is part of the new 32-response group. No old answer is merged into it.
Save the changed request order. Batch-dependent numerical trajectories can differ from v9.
Then grade the new group with the same Luna medium client and rubric controls.
The RL gate still requires 32 nonempty final answers and genuinely nonconstant rubric scores.
It now also checks the exact sampling parameters, so old-profile grades cannot qualify the repaired profile.

This uses the user's standing permission for inference failures, sampling repairs, validation and replacement GPU jobs.
Keep the original approved scientific question, model, data, prompt, reward, controls and held-out split.
One MI350, four hours, at most four GPU-hours. There are no optimizer updates in this check.
All submitted configurations and old results remain intact.

## Training implications

Prepare `rl_smoke_v5.json` against this new group and profile. Do not submit it before its own gate passes.
Keep the two-update, eight-MI350, four-hour bound and all optimizer/MTP/async settings.
The Torch backend uses more memory than the tiled head. It has not been qualified by generation.
At sequence length 98,304, vocabulary 248,320 and tensor parallelism two, one local BF16 logits tensor is 22.73 GiB.
An FP32 tensor of that shape is 45.47 GiB, and a Boolean support mask is 11.37 GiB.
These are tensor components, not a complete peak-memory estimate. MTP, backward buffers and activations add memory.
Measured GPU capacity is 287.98 GiB. Require actual finite training and memory evidence before claiming a pass.
The vocabulary is divisible by the TP2 padding unit, so this configuration adds no vocabulary-padding slots.
If training fails, record and validate a runtime repair within the same scientific plan.

CPU controls reject a missing sampled token, non-normalized support, a failed first sample, and a mismatched gate profile.
Native sampling metadata, completion of the failed seed, full-group reliability and nonconstant rewards remain unverified.

## Compatible-runtime fallback during the native queue wait

At 14:27 UTC, AMD forecast job 534420 at 11:02:30 CST, about 2.5 hours later.
The forecast is not a reservation. Use the already measured compatible MI210 image in parallel.
Fresh `group_check_v11_mi210_vllm8.json` uses the same bounded profile, fixed question and all 32 original seeds.
It retains target-model vLLM decoding and eight-request admission from the measured working capacity repair.
Validate sample 25 before the remaining 31. Save the actual sampling signature for its own gate.
This fallback does not validate SGLang support metadata. Keep the native diagnostic queued.
Once the bounded fallback passes model startup and starts real generation, stop the owned unbounded v7 job.
Preserve and grade its partial answers. Old-profile answers cannot open the repaired profile's gate.
One MI210, four-hour maximum. Do not claim success from submission or a startup log.

## Observed native feature incompatibility and target-only replacement

Job 534420 started earlier than its forecast and failed after 138 seconds.
The model warmed up. The 16-token native support control then raised:
`return_sampling_mask is not supported with speculative decoding.`
No study answer or reward was produced. The old startup and failure files remain preserved.
Immutable evidence: https://huggingface.co/datasets/3N3G/internalization-runs/tree/ace4bc6adc70b30ae4f57aaee4113c2323fd0c89/miles/group-check-v10-mi350

Fresh `config.sampling_repair_v2.json` removes only the four inference speculation fields from v1.
Target-model decoding leaves the intended target policy unchanged. Actor MTP training remains enabled.
The native engine helper still requires its capacity fields. It omits only absent optional speculation fields.
The replacement records the resolved engine mode and fails if speculation unexpectedly remains enabled.
Use `group_check_v12_mi350.json` for the same support control, failed-seed-first validation and complete group.
Use `rl_smoke_v6.json` after a qualifying gate. V5 is superseded because this image rejects its feature combination.
All model, question, rubric, cap, sampling-filter and optimizer fields remain unchanged.
Actual support capture and trainer memory remain unqualified until their checks run.

## Larger-memory compatible fallback

The compatible MI210 trial completed the original failed seed at 40,004 tokens with a natural stop.
Luna scored that answer zero in both orders. The remaining requests continue. This is not a varied group.
A read-only scheduler check then found five of eight MI300 single-GPU slots idle.
Slurm's empty GRES field does not identify those partitions as CPU-only.

Use fresh `group_check_v13_mi300_vllm32.json` with the same verified SIF, model, prompt, profile and 32 indexed seeds.
Require one visible GPU, successful FP32/BF16 kernel controls, and at least 160 GiB of measured GPU memory.
Allow 32 running requests. Their actual prompt plus 90k output cap needs 2,891,968 KV-token positions.
Check the measured model cache before retiring the slower owned trial.
The failed seed has already passed in this compatible backend. Generate a fresh full group concurrently, including that seed.
Do not stitch in its old answer. This fallback does not qualify SGLang metadata or trainer backward computation.
The one-GPU/four-hour cap remains. Record every result and preserve each old attempt.
