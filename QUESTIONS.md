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
- Status: rechecked on 2026-11-20. The same ED25519 fingerprint still fails verification.
  No saved host key was changed. Evidence: `experiments/tcs_async_rl/evidence/flame_access_20261120.json`.

## 2026-11-18: Gemini backup access

- Question: does Gemini use a direct Google API credential or a different provider endpoint?
- Options: repair direct Google access, or provide the intended endpoint configuration without credential values.
- Recommendation: validate the intended route before relying on Gemini as an independent judge.
- Blocks: backup/independent-provider validation, not the approved GPU generation or provisional Luna grading.
- Status: rechecked on 2026-11-20. Google returns HTTP 401 `UNAUTHENTICATED` using the file's Gemini assignment.
  No alternate `GOOGLE_API_KEY` assignment exists. No Gemini generation request was sent.
  Evidence: `experiments/tcs_async_rl/evidence/gemini_backup_diagnostic_20261120.json`.

## 2026-11-20: W&B authentication for runtime logs

- Question: where is the user's existing W&B credential loader or login?
- Evidence: no W&B environment assignment or default login was found on the VM or AMD.
  The user-specified provider file has no W&B assignment name. No credential values were inspected.
- Options: preserve native offline W&B files, or sync from the VM through an existing authenticated loader.
- Recommendation: keep offline W&B and TensorBoard now. Sync later if a permitted loader is available.
- Blocks: an online W&B URL, not the authorized rollout or RL runtime checks.
- Status: asked in chat on 2026-11-20. No key value was requested.

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

- 2026-11-20: user authorized continued rollout repair until non-constant real rubric rewards,
  then RL submission and debugging until a run works. The first bounded runtime plan is
  `experiments/tcs_async_rl/RL_SMOKE_PLAN.md`: two updates on the same diagnostic question,
  one eight-GPU node for at most four hours, and a separate $5 judge cap.
  The full-corpus development split and learning horizon remain undecided.
- 2026-11-20: use the already authorized VM credential loader for online Luna scoring during
  the requested RL run. The provider client remains on the VM, reached through a loopback SSH tunnel.
  No provider key is copied to AMD or placed in Ray job arguments.
