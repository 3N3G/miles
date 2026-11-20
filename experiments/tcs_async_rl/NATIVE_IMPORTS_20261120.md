# Native training import diagnostic

The compatible vLLM image is generating the fixed-policy group in job 534098.
It lacks the SGLang, Megatron, Ray and Transformer Engine training stack.
The original native image failed GPU kernels on MI210. Its MI350 check remains queued as 534034.
No native training launcher has run yet.

Before allocating eight GPUs, run `python3 train_async.py --help` in the original native image.
Use the already cached pinned image and a fresh detached checkout and run directory.
`native_imports_v1.json` bounds this diagnostic to one MI210 for ten minutes.
The launcher imports its native dependencies and builds argument help. It starts no Ray cluster or training.
It reads no private question, calls no judge, and changes no scientific setting.
Keep all output and failures. A pass qualifies imports and argument availability only.
It does not qualify GPU kernels, model loading, backward computation, or MI350 compatibility.

This is an authorized deployment diagnostic under the user's instruction to debug the approved RL run.
The RL submission gate still requires a complete group with nonconstant real rubric rewards.

## First result and exact repair

Job 534132 failed after 54 seconds. It reached no argument parser or GPU training.
`deepseek.py` eagerly imported SGLang `encoding_dsv4`, absent from the pinned image.
Qwen uses its Hugging Face template and needs neither DeepSeek V4 nor V4.1 encoders.
Load those optional encoder modules only when their family is selected.
Keep the same encoders and all existing DeepSeek rendering logic when available.
An unavailable selected family still raises an error. There is no replacement template.
The regression test imports the bridge with the optional encoders blocked, detects Qwen,
and verifies both unsupported selected families fail. CPU tests and style checks pass.
Repeat the same bounded native help check from `native_imports_v2.json` and a fresh checkout.
Do not change either active generation job. Full native GPU/runtime compatibility remains unqualified.

Private first-attempt results:
https://huggingface.co/datasets/3N3G/internalization-runs/tree/b5396bea8f0b28c332cbe1c5a927e7b7a6012686/miles/native-imports-v1

## Repaired result

Job 534140 completed successfully in 66 seconds. Native `train_async.py --help` returned zero.
This validates the repaired import path against the actual pinned container.
Miles suppresses Megatron argument help. Missing Megatron flags in this output are not evidence of unsupported arguments.
The source includes the planned FP32-gradient flag. Exact full argument validation and GPU execution remain untested.
Private results: https://huggingface.co/datasets/3N3G/internalization-runs/tree/10a47765916dfc157493779b884017272e99f302/miles/native-imports-v2

## Exact argument and process-isolation check

The help check does not execute native argument validation or test the required private process namespace.
Run `native_imports_v3.json` on one MI210 for at most ten minutes.
Use the same container and actual model arguments from the framework model script.
Resolve the authorized bounded run snapshot `rl_smoke_v2.json`, then call the native parser.
The v2 RL snapshot changes only diagnostic gate/input paths and its job name. It changes no scientific setting.
This diagnostic explicitly returns before any conversion, Ray launcher, reward call, or optimizer update.
It verifies a different PID namespace from the host before parsing. It reads public model configuration only.
If native validation requires the not-yet-converted checkpoint or a GPU kernel, record that boundary as unqualified.
Do not fabricate a checkpoint or bypass a native validation requirement to claim success.
The regular RL path retains its full group gate, private namespace guard and eight-GPU count check.

## Native-registration failure and repair

Job 534162 reached the full parser after its private PID namespace guard passed.
It failed in 67 seconds because the alias loader raises whenever Transformers gains native support
for an unrelated alias. Transformers 5.19.0 now supports `qwen4_exp_text`.
The requested model is Qwen3.5, so this conflict should not prevent its startup.
Preserve native Transformers support unless the alias explicitly sets `override_hf_native=True`.
Existing explicit DeepSeek overrides retain their prior behavior. Do not replace native Qwen4 configuration.
The focused regression verifies unrelated model loading, repeated calls and no alias replacement.
It passed on the VM with Transformers 5.19.0. Broader fixture loading needs Torch, absent on the VM.
The next job runs both native-preservation and explicit-override tests in the real image,
then repeats the exact argument/PID check from `native_imports_v4.json`. No training is launched.
Private failed-attempt results: https://huggingface.co/datasets/3N3G/internalization-runs/tree/1c7477f54075c579f43b57c13f824194d09c1838/miles/native-imports-v3

## Regression-scope correction

Job 534165 stopped in the focused test after 27 seconds.
The explicit native override test passed. The new preservation test incorrectly rejected
SGLang registering its own unrelated ASR/OCR configurations during an optional import.
Restrict the assertion to the alias under test. Keep the no-fallback-import assertion.
The corrected focused test passed on the VM. Repeat both tests and exact argument validation
from `native_imports_v5.json`. Preserve the failed test output at https://huggingface.co/datasets/3N3G/internalization-runs/tree/4c31ef170a421284e78c967c6ed04806bf4e847a/miles/native-imports-v4

## Communication-environment repair

Job 534173 passed both native-alias regression tests. Full argument validation then required
`CUDA_DEVICE_MAX_CONNECTIONS=1`; the diagnostic exited after 87 seconds without training.
The framework owns this Megatron default in `command_utils.common.train_env_vars`.
Ray training receives it automatically. The converter launches before Ray and needs it too.
Call that existing helper before conversion and diagnostic validation. Do not duplicate its constants
or change communication/model parallelism. Repeat exact validation from `native_imports_v6.json`.
The helper receives the Megatron request fields it reads and the same ExecuteTrainConfig type as training.
Private attempt results: https://huggingface.co/datasets/3N3G/internalization-runs/tree/4098ab12f2d0269267332f69a3ad595f76277bd6/miles/native-imports-v5

## Successful native argument qualification

Job 534191 completed in 91 seconds. Both alias regression tests passed.
The exact native model/run arguments and private PID namespace passed validation.
The parsed values preserve fully async CISPO/GRPO, 32 responses, global batch 32, two rollouts,
90,000 generated / 98,304 context tokens, TP2/PP1/CP1, MTP1, and two actor/six rollout GPUs.
The parser maps the negative W&B suffix flag to `wandb_random_suffix=False`.
It resolves a missing run checkpoint to the planned reference-conversion path as intended.
The reference conversion has not run or been fabricated. No model loading, Ray cluster or optimizer ran.
Private evidence: https://huggingface.co/datasets/3N3G/internalization-runs/tree/203dd98c8a6bd85fe51f3c1885257629f1c04d01/miles/native-imports-v6

The ready runtime snapshot is now `rl_smoke_v3.json`. It differs from the validated v2 snapshot
only in job name and diagnostic input/gate paths, which now refer to the eight-request group.
A whole-config comparison verified that all other fields match. Its RL gate is still closed.
