# Four-problem continuation: operations

The user confirmed that “Do it” approves the concrete run and resource choices discussed.
The approved run has ten new updates, four hours on eight MI350 GPUs, and a separate $5 judge ledger.
The earlier ledger remains unchanged. All original checkpoints remain preserved because AMD now has sufficient space.

## Submission and immutable source

- AMD job: **544406**, `tcs-easy-overfit-v17`, user `geney`, allocation `k007-004`.
- Submitted: **2026-11-24 01:15:23 UTC**. Started: **01:15:40 UTC**.
- Source: **`ae2ba558539c15188c19b3c5c112acaffee4210f`**.
- Configuration: `rl_easy_overfit_v17.json` at that source commit. The earlier proposed JSON remains unchanged.
- Resume: v16 iteration **98**, containing 99 saved optimizer updates.
- Target: global updates **99–108**, with four problem groups and 128 consumed answers per update.
- Requeue is disabled after checking scheduler owner and job name.
- No source checkpoint or old result was deleted.

AMD run: `/work1/aviralkumar/geney/tcs-async-rl/runs/easy-overfit-v17`.
AMD frozen checkout: `/work1/aviralkumar/geney/tcs-async-rl/checkouts/easy-overfit-v17`.
VM run: `/home/ubuntu/.cache/miles/tcs-async-rl/runs/easy-overfit-v17`.
VM controller checkout: `/home/ubuntu/.cache/miles/tcs-async-rl/checkouts/easy-overfit-controller-v17`.

The normal `auto_rl` controller verified the unchanged diagnostic gate and all 60 question-specific controls.
Its 120 cached provider receipts match exact payload hashes, judge model, effort, prompt and expected rubric outcomes.
It made no new control requests. `control-cache-provenance.json` preserves the source receipts and hashes.

## Durable artifacts and bounded staging

The private Hugging Face dataset `3N3G/internalization-runs` owns durable artifacts.
Existing authentication verifies both read and write access. No new key or credential transfer is needed.
Raw outputs, consumed batches, grade receipts, metrics and plots use `push-run miles`.
Checkpoints remain on AMD. The VM retains operational metadata and bounded temporary staging.

The 128-answer batch can produce a compressed trace above the earlier 64-MiB transfer target.
The transfer now permits one oversized atomic file at a time if it fits the one-GiB VM reserve.
After upload, the verifier checks each immutable Hub LFS SHA256 or Git blob hash and file size.
Only a verified trace is indexed and removed from VM staging. Upload or verification failure preserves both copies.
All 117 TCS tests pass, including oversized transfer, wrong remote hashes and local-retention controls.
The verifier also passed a real 26,851,630-byte trace readback.

Preflight archive: **`b769323264180c4feac67b42380952f02d587d66`**, folder `miles/easy-overfit-v17-preflight`.
Five immutable file readbacks verify its approved config, cached-control provenance, diagnostic gate,
actual Hub-hash control and preparation receipt. The receipt remains under that preflight's `archive-readbacks/`.

## Supervision

Session UUID: `01a1ed24-1032-7731-910b-824f484f5f71`, VM `home`, original directory `/home/ubuntu/miles`.
Owned VM process receipts record controller **1382535**, reward service **1382542**,
metrics observer **1382881**, GPU telemetry **1382882**, and hourly review **1382883**.
Verify the receipts and current process identity before managing a process.

Watch through terminal scheduler state, checkpoint inspection, immutable uploads and W&B readback.
Inspect optimizer progress, finite metrics, actual four-by-32 batch shape, within-problem reward variation,
entropy, divergence, gradient norms, update direction, engine publications, policy lag, provider failures and both disks.
Read actual proofs and both grading decisions at least hourly. A startup success does not finish supervision.

Keep the single-checkpoint policy. If unfinished with 30 minutes left, or judge spend plus reservations reaches $4,
record the intent and write this run's `checkpoint-request` sentinel once after owner/name validation.
Finish the active batch, verify the checkpoint tracker/files and its actual three-engine weight publication,
then stop only the owned allocation if the save was early. Never fabricate a reward after a budget rejection.
The remaining dollar is reserved for larger pending batches. Do not reset the ledger or enlarge any limit.

W&B uses one VM-native history writer in `geney/tcs-async-rl`, with explicit optimizer coordinates.
The independent observer keeps metrics and no-history heartbeat refreshes moving during artifact uploads.
Do not merge the separate AMD offline streams into that history.

## Current qualification

The trainer loaded iteration 98. Both live optimizer ranks report scheduler counter 3,168 and constant learning rate 1e-6.
All six loaded worker-module hashes match the frozen source. The restored initial three-engine publication
matches all 822 language-tensor checksums of the saved v16 iteration-98 publication exactly.
The observer uses Ray Client loopback after two external-driver node-discovery attempts failed.
Those attempts and their logs remain preserved. The training process was unaffected.

The first live archive **`e3db6bf2b484490defd9c71e9c358db69e043336`** verifies 42 generated-output files,
17,338,368 bytes, before VM eviction. The live `archive-index.json` and per-commit `archive-readbacks/` own subsequent counts.
Fourteen older duplicate v16 audit traces also passed immutable Hub-hash verification before VM-cache eviction.
That freed 296,111,666 bytes. Their AMD originals and every checkpoint remain intact.

## First complete update

Global update **99** completed normally at **2026-11-24 01:49:16 UTC**.
The exact payload contains four separate 32-answer groups. All four have within-problem reward variation.
Dynamic scheduling accumulated **66 microbatches** before one optimizer step.
The three post-update engines agree, with **458 of 822** language-tensor checksums changed from the restored policy.

Its raw mean reward is **6.359375/7**, with 127/128 nonzero answers and 72/128 full-credit answers.
This first batch was generated entirely with restored weight version 99, before the new optimizer update.
It is a startup baseline for this continuation, not an observed learning gain.
Gradient norm is **0.0994657**, full-policy entropy **0.426614**, sampling entropy **0.354452**,
sampled initial-policy divergence **0.00948196**, and rollout/trainer divergence **0.000256076**.
Per-token effective sample ratio is **0.9994875**, clip fraction zero, and learning rate remains **1e-6**.
All observed metrics are finite. These checks do not establish mathematical proof accuracy or reduced gradient variance.

The [new W&B run](https://forge.coreweave.com/wandb/geney/tcs-async-rl/runs/261124-011641-925-native)
has one unique history row at optimizer coordinate 99. All **80** scalar values match native logs by fresh API readback.
The 78,627,642-byte consumed trace matches its immutable Hub SHA256 in
**`299ba1ea7926f872456abad5bbb4e955e469f27c`** before VM staging eviction.
`first-update-verification.json` and `first-update-all-wandb-scalars-verification.json` preserve these checks.

Four seed-selected raw answers and both original grades are preserved for inspection.
They include a valid cut-packing argument, an answer with incorrect clipping case equalities,
a valid vacuous proof of the supplied graph statement, and a Newton-basis proof with unresolved rubric-boundary deductions.
The assistant review is separate from human calibration and changes no reward.

Training continues within the approved limits. The second 128-answer payload is ready.
