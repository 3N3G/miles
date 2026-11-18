# Open questions

## 2026-11-18: Output-head precision

- Question: enable FP32 logits on both trainer and rollout engine for learning?
- Options: qualify and select matched FP32 logits, or retain model-dtype logits after measuring numerical mismatch.
- Recommendation: test existing FP32 paths on AMD and compare fixed-weight probabilities before selecting.
- Blocks: final precision flags. FP32 is implemented but not selected by the current proposal.
- Status: evidence and existing GPU checks are documented in `experiments/tcs_async_rl/RUNTIME_READINESS.md`.

## 2026-11-18: Thinking text passed to the grader

- Question: grade only the final answer, or also the generated thinking text?
- Options: final answer only, or full generated text.
- Recommendation: final answer only, with explicit handling of unfinished thinking and truncated output.
- Blocks: the production adapter's extraction policy and calibration packet.
- Status: the approved diagnostic extracts the final answer and leaves unfinished thinking ungraded.
  The online Miles adapter still needs review before learning.

## 2026-11-18: Pilot resource limits

- Question: what GPU-hour and judge-API spending caps should the pilot use?
- Options: a bounded technical smoke test, or a bounded smoke test plus learning pilot.
- Recommendation: approve a short runtime check before a learning allocation.
- Concrete first proposal: one MI350 eight-GPU node for two hours maximum, zero judge calls and no learning.
  See `experiments/tcs_async_rl/QUALIFICATION_PLAN.md`. This does not select a learning horizon or spending cap.
- Blocks: exact jobs, walltimes, sample counts, judge concurrency, and paid calls.
- Status: the user approved the separate one-prompt inference check. Its one-GPU, four-hour job is submitted.
  The eight-GPU technical qualification and full learning allocation remain separate proposals.

## 2026-11-18: Development split

- Question: reserve which eligible papers for development?
- Options: reserve development papers within the ten eligible cohorts, or use training diagnostics only.
- Recommendation: reserve development papers. Review a small human-labeled judge packet before training.
  Use a second provider for independent assessment. Do not infer proof correctness from judge agreement.
- Blocks: the experiment's development manifest and checkpoint-selection rule.
- Status: the supplied seven-point rubrics and Luna/Gemini provider roles are selected. No development split is implemented.

## 2026-11-18: Flame host key after repair

- Question: does the repaired login host have ED25519 fingerprint
  `SHA256:A7VaRWl2lzXVUdArY/+N+Ob9K+Q0hzomXTFlyqC7x8A`?
- Evidence: the Cloud API reports the login VM RUNNING. IAP reaches its SSH server, but host-key verification fails.
  The authenticated Cloud API denies guest-attribute and serial-output access, so those checks cannot verify the key.
- Options: verify the fingerprint through an administrator or trusted console and update this host's entry,
  or leave the saved entry unchanged while using AMD.
- Recommendation: verify the fingerprint before updating the saved host key. Keep strict host checking enabled.
- Blocks: reading current Flame GPU inventory and using its login host. AMD preparation continues.
- Status: asked in chat. No saved host key was changed.

## 2026-11-18: Runtime judge credential route

- Question: how should the AMD training workers reach a permitted judge client?
- Evidence: the provider assignment names exist in the VM credential file. That file is absent on AMD.
  The user authorized VM-only loading for the group diagnostic. Luna metadata and six control calls succeeded.
  Gemini metadata returned HTTP 401. No credential was transferred to AMD.
- Options: a user-configured environment loader on AMD, or a judge service on the VM that owns provider access.
- Recommendation: choose the existing permitted loader if one is available. Keep key values out of logs and Ray job JSON.
- Blocks: online judging during learning. Post-hoc diagnostic grading on the VM works.
- Status: the production route remains unresolved. The approved local loader does not deploy a VM judge service.

## 2026-11-18: Gemini backup access

- Question: does Gemini use a direct Google API credential or a different provider endpoint?
- Options: repair direct Google access, or provide the intended endpoint configuration without credential values.
- Recommendation: validate the intended route before relying on Gemini as an independent judge.
- Blocks: backup/independent-provider validation, not the approved GPU generation or provisional Luna grading.
- Status: authorized VM metadata GET returned HTTP 401. No Gemini generation request was sent.

# Answered questions

- 2026-11-18: user approved a quick AMD inference check of one group.
  Job and immutable input/config/source provenance are recorded in `experiments/tcs_async_rl/jobs.jsonl`.
  User also authorized loading the VM credential file locally and grading the group without displaying or copying values.
- 2026-11-18: enable Qwen thinking and keep Luna at medium effort. User's explicit reply.
  The pinned Qwen template exposes `enable_thinking`, without a high-effort control.
  The proposed config now sets `apply_chat_template_kwargs={"enable_thinking": true}`.
- 2026-11-18: hold out FOCS 2026 and STOC 2026. User's explicit instruction.
  SODA 2026 remains eligible. No held-out Parquet shard was downloaded.
- 2026-11-18: Ian's command is a usage example. Replace settings that do not fit TCS.
  The reference command does not freeze batch sizes, topology, hooks, or experiment-specific settings.
- 2026-11-18: the user created `3N3G/miles`. GitHub metadata confirms its parent is `IanYHWu/miles`.
  Use SSH for VM writes and HTTPS for AMD reads. The user registered the VM's dedicated deploy key.
  SSH read/write access is verified. Preparation commits are pushed and the AMD checkout is synchronized.
  Keep private dataset examples outside the public source repository.
- 2026-11-18: use a 90,000-token solver output cap. User said, "yeah lets do 90k."
  Use the matching planned total context of 98,304 tokens. The judge budget remains separate.
  The 90k option came from Ian's quoted suggestion, not measured Qwen generation lengths.
  This token-budget decision does not authorize GPU submissions or paid judge calls.
- 2026-11-18: work backward from available GPU resources and reuse Ian's relevant settings.
  The resource-based JSON proposes two trainer GPUs, six rollout GPUs, and the original 64x32/global-2048 batch.
  This is a reviewable proposal, not a submitted training run. GPU memory/throughput still needs qualification.
- 2026-11-18: use each dataset question's existing rubric with GPT-6 Luna, and prepare a Gemini backup.
  The user authorized API-key checks. Checks used process environments only and sent no generation requests.
- 2026-11-18: judge key assignments are in `~/.config/api-keys.env` on the VM.
  The initial permission covered assignment-name checks only. Both provider names were present.
  The user later authorized the VM client to load those values locally and grade the group.
  Luna access works. Gemini metadata returned HTTP 401. No credential value was displayed or transferred to AMD.
