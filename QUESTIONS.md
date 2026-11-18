# Open questions

## 2026-11-18: TCS thinking mode

- Question: should Qwen use thinking mode? Ian's reference command disables it.
- Options: enable thinking, or preserve the reference's disabled mode.
- Recommendation: enable it for the proposed reasoning pilot, subject to the user's decision.
- Blocks: the exact solver prompt and the rule for which answer text the judge sees.
- Status: asked in chat. Both tokenizer template previews are saved locally.

## 2026-11-18: Pilot resource limits

- Question: what GPU-hour and judge-API spending caps should the pilot use?
- Options: a bounded technical smoke test, or a bounded smoke test plus learning pilot.
- Recommendation: approve a short runtime check before a learning allocation.
- Blocks: exact jobs, walltimes, sample counts, judge concurrency, and paid calls.
- Status: asked in chat. No GPU job or paid API call has been submitted.

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

# Answered questions

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
  The user authorized checking existence, not values. Both `OPENAI_API_KEY` and `GEMINI_API_KEY` assignments exist.
  Values were not parsed, extracted, displayed, or copied. No API request used this file's credentials.
  The earlier invalid Gemini result applies to AMD's process environment, not this file. Provider validity is unknown.
