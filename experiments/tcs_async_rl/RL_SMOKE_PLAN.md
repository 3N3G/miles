# Experiment: two async CISPO optimizer updates on the diagnostic TCS question

Status: authorized by the user on 2026-11-20 after the rollout check produces non-constant rewards.
The user asked to submit and debug the RL run until it works, and use their W&B account where possible.
This plan bounds the first training qualification. It does not select the full-dataset learning horizon.

## TLDR

Run two optimizer updates on the same eligible question used by the frozen-model diagnostic.
Each update consumes one group of 32 responses. Keep the approved model, prompt, rubric, thinking, and token limits.
Use fully asynchronous generation and CISPO, the clipped importance-sampling policy optimizer from Ian's example.
Require actual varied rubric grades before submission. Preserve every failed attempt and its configuration.

`rl_smoke_v3.json` binds the vLLM group to its grade file.
The prepared `rl_smoke_v4.json` instead binds native MI350 group v9 to its grade file.
These snapshots differ only in run identity and diagnostic gate provenance. Their training settings are identical.
Choose a snapshot only after its own complete group meets the same measured reward gate.
The private input SHA-256 is identical for both groups. No RL allocation has been submitted.

After the native repetition failure, `rl_smoke_v5.json` binds the replacement v10 group and its new sampling profile.
The exact failure, validation order and memory implications are in [the sampling repair](REPETITION_REPAIR_20261120.md).
The replacement uses the existing Torch log-probability backend because bounded sampling requires support replay.
Old-profile grades cannot open its gate. No reward definition, question or judge changes.

## Background / motivation

Generation alone does not validate backward computation, optimizer state, weight publication, or resumable saving.
Two updates exercise repeated training, saving and weight publication.
The asynchronous producer can finish samples before publication or continue them across versions.
Report the versions actually observed in both consumed groups. Two updates do not guarantee fresh-weight samples.
It reduces deployment uncertainty. It cannot estimate learning quality or generalization.

## Concrete question we are answering

Can this AMD deployment finish two optimizer updates with finite training statistics and saved checkpoints?
If yes, the runtime is qualified at the observed lengths and concurrency.
If no, preserve evidence, identify the failing component, repair it, and rerun the affected check.
Success does not prove that every 90,000-token response fits or that corpus-scale training is ready.

The most likely early failures are incompatible container libraries, checkpoint conversion, model kernels,
long-sequence memory use, judge transport, and trainer-to-generator weight transfer.
The existing frozen-model group and passing reference/empty/irrelevant judge controls precede training.

## Methodology

- Model and data revisions: inherit `config.proposed.json` through the frozen `rl_smoke_v1.json` run snapshot.
- Data: the same single SODA 2023 polygon-packing question from `GROUP_CHECK_PLAN.md`.
  For example, it asks for constructive strip packing with approximation bound 51 and related guarantees.
  Its six rubric items have weights 2, 1, 1, 1, 1, 1. Private artifacts contain the complete input and reference.
  FOCS 2026 and STOC 2026 remain sealed. No new questions are selected for this check.
- Generator: Qwen3.5-4B with thinking enabled, 90,000 output-token ceiling and 98,304 total context.
  Temperature one, top-p one, no top-k cutoff. Keep terminal/thinking markers so extraction matches the diagnostic.
- Runtime: one eight-GPU MI350 node, two trainer GPUs and six rollout GPUs in three engines.
  Tensor parallelism two, full activation recomputation, the snapshot's head backend, BF16 parameters,
  FP32 accumulated gradients, and the existing MTP/speculative settings from the proposal.
- Algorithm: inherit CISPO/GRPO settings and Adam learning rate from the proposal.
  Bound this runtime check to one prompt per rollout, 32 samples per prompt, global batch 32,
  and two rollouts. These batch limits replace the unqualified corpus-scale 64/2,048 proposal for this check.
  Keep the async buffer factor and staleness bound. Save after each update.
- Reward: the supplied weighted seven-point rubric. Luna medium judges the final answer in both anonymous orders.
  Average the two candidate scores. The VM owns provider credentials and private judge receipts.
  AMD calls a loopback endpoint through an SSH reverse tunnel. Failed/incomplete judgments stop the run.
  A recorded model token limit can leave an empty final answer. The same judge grades that empty answer in both orders.
  Both empty verdicts must be zero. Unexpected incomplete outputs, provider failures and parser failures stop the run.
  Record model truncation separately. The diagnostic gate still requires 32 nonempty final answers.
  No old diagnostic response is rescored. See [the finish-handling repair](REWARD_FINISH_HANDLING_REPAIR_20261120.md).
- Controls: reuse the passing diagnostic controls. Submission requires all 32 diagnostic final answers graded,
  at least two measured reward values, and reference/empty/irrelevant scores 7/0/0.
  Human calibration and an independent-provider audit remain pending. All rewards are provisional.
- Metrics: completed rollout groups, reward distribution and answer-order gaps, response lengths,
  finite loss and gradient norm, skipped optimizer steps, optimizer iteration, weight versions,
  checkpoint tracker, elapsed time, peak memory where available, and API usage/cost.
- Seeds: 20261118. This is one runtime check, with no three-seed learning-effect claim.
- Statistical test: none. Success is a runtime acceptance check, not a hypothesis test about improvement.
- Plots: provisional reward distribution and response-length distribution for each consumed group.
  Show one seed-selected raw response with its per-item judge decisions next to the aggregates.
- Logging: durable stage logs, native rollout metrics, raw rollout snapshots, private judge receipts,
  TensorBoard, and W&B offline files. No W&B credential was found on either machine.
  Sync to the user's account from the VM if an authenticated loader becomes available.
- Resource cap: four hours on eight GPUs, at most 32 GPU-hours for one attempt. Judge spending cap $5.
  Cache the recipe-specific initial conversion under the model directory. Preserve failed runs and checkpoints.

## Runtime acceptance checks

1. The submitted source, resolved native arguments, model/input revisions, and gate evidence are saved.
2. Two rollout groups reach the trainer with real rubric grades and finite statistics.
3. The optimizer completes two updates without a skipped/non-finite step.
4. Initial and subsequent weight publications succeed. The initial equality check passes.
   Report consumed per-token weight versions and any observed generation under updated weights.
   Do not infer fresh-weight generation solely from the update count.
5. The final checkpoint and rollout state are durable, and the launcher exits successfully.
6. Logs and descriptive results are uploaded to the private results dataset. W&B sync status is explicit.

## Results

Not submitted. The frozen-model diagnostic gate has not passed yet.

## Global takeaways

Reserved for the user's interpretation. This check cannot establish learning or proof correctness.

## Next steps

1. Complete the diagnostic and inspect its raw scores and controls.
2. Start the local reward service, validate an actual reference call, then submit the committed RL job.
3. Attach and verify the tunnel while the allocation starts. Monitor generation, training, saving, and exit.
4. Record repairs before replacements. Preserve each submitted configuration byte-for-byte.

## Questions

Full-corpus development split, learning horizon, independent judge access, and human calibration remain open.
They do not turn this runtime check into a scientific evaluation.
