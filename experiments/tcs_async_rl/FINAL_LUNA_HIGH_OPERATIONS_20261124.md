# Final Luna-high comparison execution record

- Session: `01a1ed24-1032-7731-910b-824f484f5f71`, machine`home`, original directory`/home/ubuntu/miles`.
- Plan: [third and final additional prompt candidate](JUDGE_ITEM_REPAIR_20261124.md).
- Config: [judge_comparison_items_high_v1.json](judge_comparison_items_high_v1.json).
- Frozen labels committed and pushed at`bdd61e7c`, before any new request.
- Packet SHA256:`d6f3df5dc72b4e4007ccab06d56c6665c263f09e4db239a1d336fde57a3fb16d`.
- Label SHA256:`2b4d1966b295b332d4447fe53f423f3b83ba45839dd2f69b3385811d16ecf277`.
- VM artifact owner:`/home/ubuntu/.cache/miles/tcs-async-rl/runs/judge-comparison-item-v5-20261124`.
- Exact entry point:`python -m experiments.tcs_async_rl.compare_reference_prompts` with this frozen packet, labels, config and output directory.
- 81 paid candidate attempts for 80 valid presentations, including one malformed-verdict retry. No valid score rerolls.
- 132 control calls. All 213 new calls settled, with no outstanding reservations.
- The exact incoming ledger contained 1448 settled calls. Its transfer receipt incorrectly recorded 1382.
  A separate count-correction record preserves the original receipt. The ledger bytes and $1.5717538 incoming cost were correct.
- The final ledger contains 1661 settled calls and $1.7725486 usage.
  Comparison spending, including the conservative earlier comparison/control prefix, is $1.0938164 within its $2 sublimit.
- Both candidates failed the unchanged gate. No selected runtime config, saved-screen regrade or new RL job was created.
- The completed screen remains archived at`3N3G/internalization-runs@7a258e0947401b05d2207f63592bea799e9284bd/miles/difficulty-screen-high-v3`.
  All 2332 files and 649,063,262 bytes were verified. The screen has 384 answers and zero optimizer updates.
- A blinded human-review packet contains all six final-prompt disagreements and the original rubric items.
  Its cases are deliberately selected disagreements, not a random accuracy sample.
- The final comparison's plot starts at zero. Its intervals resample 22 question clusters 5000 times with seed 20261124.
- W&B uses a separate judge-validation run, never the training run's history counter.
  [Comparison run](https://forge.coreweave.com/wandb/geney/tcs-async-rl/runs/judge-item-v5-20261124).
- A first W&B client invocation failed before importing the repo. The corrected invocation supplied the repository on`PYTHONPATH`.
  No remote run or judge call was created by that import failure.
- W&B finished. All 15 reported scalars passed API readback.
- Independent history readback found exactly one validation row at`_step=0`, with85% agreement and zero optimizer updates.
  The [history receipt](evidence/judge-comparison-item-v5-20261124-wandb-history.json) records that distinction from training progress.
- The final private archive is`3N3G/internalization-runs@d4d68e07f0de4ebe2c91597d8eff2b51d608a75c/miles/judge-comparison-item-v5-20261124`.
  Every one of its 511 manifested files, totaling 13,229,492 bytes, passed immutable hash verification.
  The [receipt](evidence/judge-comparison-item-v5-20261124-archive.json) records the exact commit.
  The archived report predates this receipt paragraph. No local artifact was evicted.
