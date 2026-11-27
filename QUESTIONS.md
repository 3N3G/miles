# Open questions

## 2026-11-24: Implicit monotonicity in an auction proof

- Question: does the auction payment item require an explicit single-bidder monotonicity argument when the candidate states the correct formula?
- Evidence: live answer`online-60c80dfdae4e9ed725022908` receives5/7. Its payment formulas and final cancellation are correct. It omits the monotonicity explanation and makes an unused false claim that every valid allocation covers all items.
- Options: accept the formula as a direct elementary consequence of the supplied assumptions, or require the named monotonicity step for this two-point item.
- Recommendation: mark this case ambiguous. The qualified prompt accepts direct elementary consequences, while the original rubric names the monotonicity step. Preserve its actual grade and do not count it alone as genuine mathematical variation.
- Blocks: interpreting this particular two-point score difference as a proof-quality difference. It does not block the approved fixed-grader screen or an inclusion decision supported by unambiguous medium examples.
- Permission handling: prompt wording changes remain authorized, but one ambiguous example does not itself invalidate the passing frozen-answer gate. Keep the current recipe fixed while the full screen is inspected.

## 2026-11-22: Storage for a bounded continuation after the two-hour backfill slot

- Question: may the replacement's intermediate checkpoint be removed only after a newer continuation checkpoint is atomically saved and verified?
- Evidence: the 23:13 UTC snapshot has 95.7 GB shared free space. Each checkpoint is 60.6 GB. A separate continuation would need two new checkpoint copies during a save while keeping its source, creating a 25.4 GB shortfall before a reserve. The current job fits its own save peaks.
- Options: retain original protected checkpoints and all results, but remove only the replacement source checkpoint after the first newer checkpoint is verified; or preserve every job's source checkpoint and wait for more filesystem space.
- Recommendation: allow the former under the existing one-latest-checkpoint policy for this logical continuation. Preserve original iteration 64, the earlier protected checkpoint, every used config, metric and rollout.
- Blocks: only an additional remaining-budget allocation if space stays below its safe peak. It does not block the current running job.
- Status: withdrawn at 23:54 UTC. `experiments/tcs_async_rl/CHECKPOINT_STORAGE_REPAIR_20261122.md` validates one final-or-deadline checkpoint without deleting any source. No user answer or deletion approval is inferred.
- Permission basis: the approved `experiments/tcs_async_rl/JUDGE_REPAIR_RESUME_PLAN_20261122.md` says “Preserve all old prompts, configurations, grades, and checkpoints.” Deleting this intermediate source requires an explicit exception.

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
  Rechecked on 2026-11-22 while AMD job 538647 was queued. The same mismatch remains.
  Asked for a trusted fingerprint check before inspecting fallback capacity. AMD supervision continues independently.
  Evidence: `experiments/tcs_async_rl/evidence/flame_fallback_reachability_20261122.json`.

## 2026-11-18: Gemini backup access

- Question: does Gemini use a direct Google API credential or a different provider endpoint?
- Options: repair direct Google access, or provide the intended endpoint configuration without credential values.
- Recommendation: validate the intended route before relying on Gemini as an independent judge.
- Blocks: backup/independent-provider validation, not the approved GPU generation or provisional Luna grading.
- Status: rechecked on 2026-11-20. Google returns HTTP 401 `UNAUTHENTICATED` using the file's Gemini assignment.
  No alternate `GOOGLE_API_KEY` assignment exists. No Gemini generation request was sent.
  Evidence: `experiments/tcs_async_rl/evidence/gemini_backup_diagnostic_20261120.json`.

## 2026-11-20: Judge calibration evidence

- Question: what human scores should be assigned to the saved calibration examples before interpreting learning results?
- Evidence: three saved calls gave the same supplied reference scores below seven. Independent local checks
  reject their stated geometric or algebraic objections. Original candidate scores remain unchanged.
- Options: human-label the saved packet and audit independent-provider agreement, or retain provisional runtime-only grading.
- Recommendation: complete calibration before drawing learning conclusions. Inspect paired-reference verdicts and awarded candidate criteria.
- Blocks: interpreting rewards as proof correctness and making learning claims. This does not add a new approval requirement
  to the already authorized provisional two-update runtime check.
- Status: the read-only receipt audit is in `experiments/tcs_async_rl/evidence/judge_reference_audit_20261120.json`.
  No new judge calls or score corrections were made. Gemini access remains unverified.
  The 2026-11-21 easy-run audit found different candidate scores between swapped calls for 215/864 consumed answers.
  Two pairs give the supplied reference six instead of seven points. These are not human agreement estimates.
  See `experiments/tcs_async_rl/RL_NOISE_AUDIT_20261121.md`. Original grades remain unchanged.
  A [new blinded packet](https://huggingface.co/datasets/3N3G/internalization-runs/blob/ae47d47f23c9bb54aa773b53f06d059389ad9870/miles/easy-overfit-v9-human-review-20261122-final/README.md)
  contains ten seeded training answers across nine questions and 43 binary rubric items.
  Both anonymous answers need labels. No new inference or judge call was used. Human agreement remains unmeasured.

## 2026-11-20: Dataset-wide RL feasibility versus software qualification

- Question: should the next proposed check measure representative reward coverage before choosing a corpus RL setup?
- Evidence at the original efficiency concern: one unique hard question had generated answers.
  Its two latest groups had constant zero scores and cost 3,410,866 generated tokens in total.
  The subsequent preselected easy question has 15 positive grades, 13 zeros and four capped missing finals.
  Medium completed with 28 zero grades and four missing finals. Its automatic RL handoff stopped.
  One question per difficulty cannot estimate coverage or determine corpus sampling weights.
- Options: retain the two-question software check, propose a representative eligible-question baseline,
  or propose a different initialization such as a stronger model or supervised warm-start.
- Recommendation: measure representative per-question reward variation and token cost with calibrated judgments first.
  A single qualifying question would establish software feasibility, not efficient corpus learning.
- Blocks: choosing the corpus-scale training setup. No new model, data, reward, token limit or GPU job is approved here.
- Status: the user raised this efficiency concern. Calculations and assumptions are in
  `experiments/tcs_async_rl/evidence/rl_signal_efficiency_20261120.json`.
  On 2026-11-20 the user asked how medium would inform the RL problem distribution.
  These checks can inform the starting difficulty and the prompt for the bounded runtime check.
  No easy/medium corpus mixture or adaptive sampling rule has been selected.
  A proposed broader screen would measure real within-question reward variation, completion, judge reliability and token cost
  on preselected questions across difficulties and papers before choosing weights. That screen is not yet approved.
  On 2026-11-21 the user chose easy-only RL. A checksum-verified recount found 20 easy questions across 17 papers,
  comprising 12 proof-writing and eight proof-strategy questions. This selects the difficulty subset.
  The later user reply selects 40 updates within four hours on eight GPUs. A separate development split remains unspecified.
  No broader screening job was submitted.

# Answered questions

- 2026-11-21: user approved option 1: actual Luna grades for recorded 90,000-token empty finals.
  Eight actual calls gave all four empty finals zero in both orders. Qualification now has 15 positive and 17 zero rewards.
  Unexpected missing finals and unfinished or invalid judge verdicts remain errors. Original diagnostic grades remain unchanged.
- 2026-11-21: user selected **40 optimizer updates, four-hour limit on eight GPUs** for all 20 easy questions.
  This targets two nominal passes and 1,280 consumed answers. Report actual question coverage and updates.
  Plan: `experiments/tcs_async_rl/EASY_OVERFIT_PLAN_20261121.md`. No separate frozen-policy evaluation was added.

- 2026-11-21: user selected easy-only RL and asked for the subset size.
  Include all 20 eligible easy questions, across 17 papers. They contain 12 proof-writing and eight proof-strategy tasks.
  FOCS 2026 and STOC 2026 remain excluded. Only the previously tested question has measured generation rewards.
  Evidence: `experiments/tcs_async_rl/evidence/easy_subset_count_20261121.json`.
  The later explicit cap-policy and training-budget answers are recorded above.

- 2026-11-20: user requested automatic RL submission and monitoring after the medium diagnostic.
  The durable VM controller is armed from frozen source. It retains the original strict reward gate and bounded two-update plan.
  It does not select a corpus mixture or change the unresolved recorded-cap policy.
  Procedure and evidence are in `experiments/tcs_async_rl/AUTO_HANDOFF.md`.

- 2026-11-20: user logged into W&B on the VM and selected organization `geney-org`.
  Normal SDK authentication succeeded. Its team `geney` resolves to that organization.
  Project creation under the organization entity failed. Creation and metadata lookup under team `geney` succeeded.
  This repo's W&B defaults now select the verified destination in `experiments/tcs_async_rl/WANDB.md`.
  Keep AMD logging offline and upload from the VM. No old runs were moved or provider credentials copied.
  Evidence: `experiments/tcs_async_rl/evidence/wandb_destination_20261120.json`.

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

- 2026-11-20: user approved one preselected easy question, followed by medium if its reward distribution is good,
  then using those results to inform the RL run. The approved sequence is in
  `experiments/tcs_async_rl/EASY_MEDIUM_CHECK_PLAN_20261120.md`.
  This resolves permission to change the diagnostic question. It does not establish corpus-wide feasibility.

## 2026-11-21: Sampling concurrency after native training qualification

- Question: after the first native optimizer update works, should a later attempt use 96 concurrent samples?
- Evidence: the approved 32-sample limit permits one 32-answer group in flight.
  The first group runs on one TP2 engine at about 4,800 generated tokens/second. Two rollout engines are idle.
  `miles/rollout/fully_async_rollout.py:_max_in_flight_groups` floors the sample limit by the group size.
- Options: finish the current bounded pilot at concurrency 32, or approve a replacement at 96 after native training works.
- Recommendation: validate backward computation first, then review 96 to keep three groups active on three engines.
  Keep the optimizer batch, data, reward, model and token limits fixed. Measure memory and actual sample use.
- Blocks: a later throughput change. The current approved pilot continues with its existing 32-sample setting.
- Status: observed during job 536360. No concurrency change or additional throughput job submitted.
  The complete run has 76% actor waiting time and visible decode logs from one engine actor.
  Only seven questions receive a second visit before the time limit. The throughput repair remains a user decision.
  The user requested continued work toward rising rewards. A concrete continuation is ready in
  `experiments/tcs_async_rl/EASY_CONTINUATION_PLAN_20261121.md` and `rl_easy_overfit_v10.draft.json`.
  It resumes the saved 25-update checkpoint for 40 additional updates within four hours on eight MI350 GPUs.
  It preserves the scientific setup and adds observations, with a separate $5 judge cap.
  Approval was requested for that bounded allocation with concurrency 96, or with the existing 32.
  New GPU submission and the concurrency choice are blocked pending that answer. CPU repairs and logging checks continue.

- Answer (2026-11-22 UTC): user approved the recommended 96-answer continuation and hourly detailed supervision.
  The approved snapshot is `experiments/tcs_async_rl/rl_easy_overfit_v10.json`.
  The user authorized repair and rerun for a diagnosed cause, while preserving the bounded scientific setup.

## 2026-11-22: Next step after the approved easy-question continuation

- Question: should the next allocation evaluate frozen checkpoints or continue training, after judge calibration?
- Evidence: the approved 40 new updates completed. Fifteen repeated question means rose from 5.291 to 5.428 out of seven.
  Eleven increased and four decreased. Mean answer length rose 18.2%. Scores differed across judge orders for 338/1,280 answers.
  These are uneven training visits from one seed, without an independent correctness estimate.
  See [the findings](experiments/tcs_async_rl/EASY_CONTINUATION_FINDINGS_20261122.md).
- Options: review the existing ten-example human packet first, approve a frozen-checkpoint comparison on training questions,
  or specify another bounded training continuation. Any evaluation must preserve the sealed FOCS/STOC 2026 split.
- Recommendation: label the existing packet first. Use its agreement results to plan an interpretable checkpoint comparison.
  No new GPU job or paid judge call is needed to review that packet.
- Blocks: the next experiment's design and compute budget. The approved continuation and its final audit are complete.
- Answer (2026-11-22): the user authorized assistant mathematical adjudication, grader-prompt repair, and resuming RL.
  The repaired prompt and diagnostic limits are recorded in `experiments/tcs_async_rl/JUDGE_PROMPT_QUALIFICATION_FINDINGS_20261122.md`.
  The continuation keeps Luna medium, the same model/data/optimizer, and the existing per-run bounds.
  This resolves the immediate next step. It does not supply human calibration labels.

## 2026-11-23: Next stage after the bounded judge-repair continuation

- Question: should the next stage first resolve human rubric labels, or extend training under a separately bounded plan?
- Evidence: [the complete saved-path comparison](experiments/tcs_async_rl/BOUNDED_CONTINUATION_RESULTS_20261123.md) has 14 repeated questions averaging 5.067 → 4.969/7. Six rise, seven fall, and one is unchanged. An unchanged runtime proof receives 2 versus 7 between judge orders. Exact rational-arithmetic justification and implicit one-child recursion remain rubric boundaries.
- Options: hand-label about ten saved cases and adjudicate these boundaries before another training variant; or approve an explicit training extension with a defined question, budget and checkpoint-storage plan.
- Recommendation: resolve human calibration first. Current paired agreement does not measure mathematical accuracy. Keep used prompts, grades and all source checkpoints intact.
- Resource record: the cost guard saved iteration 98, containing 99 total updates, then stopped. Shared estimated judge spend is $4.689/5. Another 60.6 GB checkpoint cannot fit the observed 23.3 GB free capacity while preserving its source. No new allocation was submitted after this stop.
- Blocks: further research-design or reward changes and another GPU allocation. It does not block logging repairs, archival or prompt wording repairs already authorized by the user.
- Answer, 2026-11-23: the user said “Do it” after the proposal to use four problems ×32 answers per optimizer update. The batch change and its run are approved. A new compute/judge limit and safe storage remain to resolve. Human calibration remains pending, and no reliable reward increase has been established.

## 2026-11-23: Resources for the approved four-problem optimizer batch

- Decision already made: the user approved four problem groups ×32 answers, global batch 128. Do not ask again about that batch change.
- Plan: [the concrete run and qualification checks](experiments/tcs_async_rl/MULTIPROBLEM_BATCH_PLAN_20261123.md). Ten proposed updates consume the original pilot's 1,280-answer budget.
- Budget question: approve a new four-hour allocation on eight MI350 GPUs (32 GPU-hours), with a new $5 judge allowance, or prepare without new spend?
- Budget evidence: the old shared allowance has 2,056 eight-GPU allocation seconds (4.5689 GPU-hours) and $0.3105281 left. Its ledger remains unchanged.
- Storage question: arrange at least 84 GB of extra free space while preserving all checkpoints, or explicitly permit removal of the superseded iteration-64 and iteration-84 checkpoint directories after checking retained original iteration 24 and latest iteration 98?
- Exact deletion option: `/work1/aviralkumar/geney/tcs-async-rl/runs/easy-overfit-v12/checkpoints` (60,935,586,232 bytes) and `/work1/aviralkumar/geney/tcs-async-rl/runs/easy-overfit-v15/checkpoints` (61,012,021,875 bytes). Preserve every non-checkpoint result and all used configs. Nothing has been deleted.
- Storage evidence: AMD has 7,597,572,096 free bytes. The unchanged preflight requires 90,696,450,048 bytes, a shortfall of 83,098,877,952 bytes. The whole TCS cache is only 2.9 GB.
- Recommendation: approve the bounded new allowance and arrange sufficient storage while preserving checkpoints. The explicit deletion option is available if preserving those two intermediate states is unnecessary.
- Blocks: only job submission and dependent provider spending. Config preparation, CPU tests and monitoring repairs continue.
- Permission basis: the approved previous plans preserve source checkpoints, and global instructions reserve experiment budgets and destructive actions for the user. Batch approval does not silently reset a nearly spent ledger.
- Answer, 2026-11-23: the user said “these are approved implicitly when I say do it.” The concrete ten-update, four-hour, 32-GPU-hour run and new $5 cap are approved. A fresh AMD check reports 1.27 TB free, so no source checkpoint is deleted. The user also requested durable Hugging Face storage, with bounded VM staging. Existing private Hub authentication works, so no new key is needed.
- Read-only evidence: [current scheduler, inventories and free space](experiments/tcs_async_rl/evidence/multiproblem_resources_20261123.json).

## 2026-11-23: Next step after the four-problem batch test

- Question: prioritize human rubric calibration, a frozen-checkpoint comparison, or a further bounded training continuation?
- Evidence: [the completed four-problem test](experiments/tcs_async_rl/FOUR_PROBLEM_BATCH_RESULTS_20261123.md) has 19 repeated questions averaging 5.752 → 5.776/7. Six rise and thirteen fall. The largest rise coincides with fewer empty capped answers. Judge orders disagree on 177/1,280 answers. This does not measure their mathematical error rate.
- Options: label the existing ten-example blinded packet without new compute; specify a bounded frozen-checkpoint comparison; or specify another bounded training continuation.
- Recommendation: complete human calibration before treating these provisional reward differences as proof improvement. All current-run code, grades and checkpoints remain preserved.
- Blocks: the next research design and spending beyond the completed ten-update, four-hour/$5 plan. It does not block the completed batch change, private artifact storage or routine diagnostic repairs.
- Approval handling: the user's “Do it” approved the concrete four-problem run and its discussed resource choices. Do not request that approval again.
- Answer, 2026-11-24: compare the anonymous two-order recipe with labeled-reference grading by agreement on pre-labeled saved answers. Then screen six easy and six medium questions, followed by up to twelve hours of RL with screening extra. The shared new paid-judge cap is $25. The user delegated agreement thresholds. The [approved plan](experiments/tcs_async_rl/JUDGE_COMPARISON_LONG_RUN_PLAN_20261124.md) owns the exact gates and resource limits.

## 2026-11-24: Next judge step after the two failed comparison gates

- Question: test the fixed repaired reference prompt with Luna high, or resolve the rubric interpretation on ten saved cases first?
- Evidence: fresh pre-labeled answers give paired-medium 29/40 exact agreement and reference-medium 32/40. Both fall below the declared 90% gate. All controls pass. All 444 paid calls settle at $0.4516942, and no GPU job or new solver answer ran.
- Options: [the concrete high-effort comparison](experiments/tcs_async_rl/JUDGE_NEXT_STEP_OPTIONS_20261124.md), bounded by the existing $2 comparison ledger within $25 total; or human review of the prepared interpretation packet without new compute.
- Recommendation: test high effort if another automatic diagnostic is preferred. Otherwise resolve the literal-versus-equivalent-proof interpretation before further reward tuning. Do not claim assistant agreement is human accuracy.
- Blocks: GPU screening and twelve-hour RL submission under the unsuccessful grading gate. The already approved resource limits remain valid and unused.
- Permission basis: the user previously selected “Enable Qwen thinking; keep Luna medium.” The current plan fixes that effort and stops after one failed prompt repair. Prompt changes alone do not authorize a different judge effort.
- Answer, 2026-11-24: “Ok try luna high. If that doesn’t work do some more prompt optimization.” This approves high effort and further prompt repair, superseding the old medium-only and one-repair constraints. The [current protocol](experiments/tcs_async_rl/HIGH_EFFORT_JUDGE_PLAN_20261124.md) preserves the existing budget and fresh-validation requirements.

## 2026-11-24: Judge decision after the final Luna-high prompt validation

- Question: resolve the remaining rubric boundaries by human review, or test a different judge before the approved long run?
- Evidence: the final Luna-high prompt matches 34/40 totals and 96.17% of items. All 132 controls pass. Six total scores disagree, four on audits marked uncertain before calls. The declared 90% total-score gate fails. No labels changed after verdicts.
- Options: review the six saved disagreements with their original rubric items, approve a specified stronger judge and fresh validation, or explicitly revise the reward/qualification protocol before a new experiment.
- Recommendation: human review first. Some discrepancies may be assistant audit errors. Do not lower the gate after observing this packet or use it as fresh validation again.
- Blocks: qualification of a new screen grader, the easy/medium inclusion decision, and the twelve-hour RL submission. Those compute permissions remain approved and unused for RL.
- Completed independent work: 384 screening answers are generated and privately archived. All final-comparison calls are settled at $1.7725486 across the shared $25 budget. All source checkpoints, used prompts and configs remain preserved.
- Details: [standalone findings](experiments/tcs_async_rl/FINAL_LUNA_HIGH_VALIDATION_20261124.md) and [operational evidence](experiments/tcs_async_rl/FINAL_LUNA_HIGH_OPERATIONS_20261124.md).
- Answer, 2026-11-25: the user said “oh i think thats fine actually” and requested the final prompt/setup plus W&B logging and durable transcripts. Keep Luna high and unchanged prompt v5 for review. Preserve its measured 85% agreement and the original failed numeric gate. The current step adds provenance and logging, without new inference or a GPU submission.

## 2026-11-25: Broader training corpus after the small overfit pilot

- Question: use all 236 easy and medium questions, or all 2,519 eligible questions including hard?
- Evidence: the pinned eligible corpus contains 20 easy, 216 medium and 2,283 hard questions. The six medium questions were a diagnostic sample, not the available medium pool. The old 20/26-question candidates carried the small overfit pilot forward.
- User steering: “Wdym selecting. Also there should be much more problems than this”. Broader coverage is requested, but the intended difficulty scope is not yet explicit.
- Options: all 236 easy+medium, or all 2,519 eligible questions. Keep FOCS2026 and STOC2026 sealed in either case.
- Recommendation: all 236 easy+medium for the next bounded run. The small medium screen does not measure the whole medium stratum, and hard questions remain unscreened.
- Blocks: the new training-data manifest and execution snapshot. This is a data-design decision, not a GPU-memory constraint. No existing config, dataset bundle or job was changed.
- Count sources: [pinned inventory](experiments/tcs_async_rl/PLAN.md) and [screen selection provenance](experiments/tcs_async_rl/evidence/difficulty_screen_selection_20261124.json).
- Answer, 2026-11-25: “Try all easy and medium. Think about why the current plots show reward flat. Then launch I give you permission”. Use every one of the236 easy/medium questions and preserve the established12-hour,8-GPU and shared$25 limits.

## 2026-11-26: Exact-reward RL control and compute cap

- Question: approve the prepared Qwen2.5-0.5B GSM8K reference-then-CISPO diagnostic, capped at four total eight-GPU node-hours?
- Options: approve the small-model control with one initial diagnostic seed, or first measure GSM8K headroom with the existing Qwen3.5-4B.
- Recommendation: start with the small-model control because the repo supplies a learning recipe and the 4B may be near ceiling. That ceiling has not been measured.
- Evidence: [concrete plan](experiments/tcs_async_rl/GSM8K_CONTROL_PLAN_20261126.md) and [all-row reward controls](experiments/tcs_async_rl/evidence/gsm8k_control_preparation_20261126.json).
- Blocks: new GPU submissions and the model/budget choice. CPU data preparation and validation are complete. No GPU job has been submitted.

- Answer: approved on 2026-11-26. User: “approved, run it”. One initial seed and four total eight-GPU node-hours cover the reference and conditional CISPO stages.

