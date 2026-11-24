# Judge comparison operational record

- VM: `home`, user `ubuntu`, original checkout `/home/ubuntu/miles`.
- Codex session: `01a1ed24-1032-7731-910b-824f484f5f71`.
- Workstream: `tcs-async-rl-recon`. No subagents were used.
- First comparison implementation: `c7c43eaa`. First labels were pushed at `587dc2ea` before calls.
- Repair implementation and fresh labels: `3f584c19`, pushed before validation calls.
- W&B configuration and actual microbatch-count fix: `cf5a475d`. All 123 TCS CPU tests pass.
  New trainer logging has not yet been validated on GPUs. No historical W&B run was changed.

## Frozen inputs

- First packet SHA256: `793dc92214c2b67049c2bcf0a36067f552969ae39f52fc22d1f101c8d34918bd`.
- First labels SHA256: `b57788dfd09fd808ef7de51ceb54d26c894c065ef044b80703ee7b4495ea850e`.
- Fresh packet SHA256: `888fc0f8f27146e1e25addc0862396ca499c2dd286e89e33dc2e798fe6770a30`.
- Fresh labels SHA256: `63aba9a7b191f2386b57bdb758af12b17691768c223bb2c3f77a759f4cf46dba`.
- Both packets have 40 actual saved answers and 176 rubric items, on the same 17 questions.
  Source identities and candidate-text hashes have zero overlap between packets.
- Existing reviewed sample identities were excluded. The one previously inspected medium answer was excluded.
- Preparation initially used a nonexistent medium directory. It failed without producing cases or making API calls.
  The corrected source is `medium-check-v15-mi350`; preparation now validates these files before downloading traces.

## Execution and accounting

Root: `/home/ubuntu/.cache/miles/tcs-async-rl/runs/judge-comparison-20261124`.
Comparison commands are `python -m experiments.tcs_async_rl.judge_comparison compare`.
The first configuration is `judge_comparison_v1.json`; the fresh validation uses `judge_comparison_v2.json`.
The latter adds `--budget-directory <root>/judgments-v1` to keep the original ledger owner.
No ledger was reset or given another allowance.

- First comparison: 120 base requests. All 17 question controls: 153 requests.
- Fresh comparison: 120 base requests. Fresh reference controls: 51 requests.
- Unchanged paired controls reused 102 exact fingerprinted receipts, copied without modifying the originals.
- Total: 444 reservations, 444 settlements, zero pending calls. Recorded cost: $0.4516942.
- All control presentations pass. No provider fallback, missing-verdict zero, or GPU job was used.
- Every request and response remains in the private archives. Credential values and HTTP authorization headers are absent.
- Both `selection.json` files say no selected method because neither passes the gate.

## Immutable private archives

1. `3N3G/internalization-runs@c2b0d8f9c65b9162f7b316494132116a193ba4fd/miles/judge-comparison-v1-20261124`.
   All 603 non-lock artifacts passed exact SHA256 or Git-blob readback verification.
2. `3N3G/internalization-runs@05550135e6475e53c77ca409c4a9ea936a6e036e/miles/judge-comparison-v2-20261124`.
   All 633 non-lock artifacts, totaling 15,284,114 bytes, passed exact readback verification.

The fresh archive includes original answers, pre-call labels, all verdicts, exact prompts, the settled shared-ledger snapshot,
question-cluster bootstrap intervals, PNG/SVG plots, and a ten-case interpretation packet without displayed judge scores.
The live ledger remains at `<root>/judgments-v1/budget.jsonl`; the archive copy is a closed snapshot, not a new budget.

## Closure and outstanding work

At 06:25:58 UTC, AMD `squeue -u geney` was empty. This task submitted no scheduler job.
All comparison and control CLI processes exited normally. The uploader and readback verifier exited normally.
No long-lived judge HTTP service, GPU tunnel or training monitor was started.
The checkpoint remains the protected v17 iteration 108. No new model state exists.

The initial easy/medium screen and twelve-hour RL continuation have not run.
The current research decision is in `JUDGE_NEXT_STEP_OPTIONS_20261124.md` and `QUESTIONS.md`.
A question asks whether to change the previously selected Luna effort from medium to high within the existing comparison cap.
No answer is inferred from a preselected UI option. Existing RL resource approval does not need to be requested again.
