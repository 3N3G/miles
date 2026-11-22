# Operational record

Original directory: `/home/ubuntu/miles`.
Current project: `/home/ubuntu/inkling-interp`.
Agent: Codex, session `01a1fd03-2b6c-7151-9835-2cb592949d73`, machine `home`.

The user's 2026-11-21 instruction requested exploratory Inkling research and allowed CPU, AMD, or Babel use.
`PLAN.md` bounds the initial diagnostic. No model training is authorized by that plan.

The existing Miles checkout has unrelated edits. They were not modified.
The short-lived Miles workstream now records relocation only.

Git storage uses the separate `inkling-interp` branch of the user's existing `3N3G/miles` fork until a dedicated remote is configured.
The project has an independent history and directory. It does not depend on Miles code.
Cluster code must come from that pushed branch. Record each submission in `jobs.jsonl` immediately.

All caches and outputs on AMD belong beneath `/work1/aviralkumar/$USER/inkling-interp-data`.
Use the existing vLLM ROCm container for PyTorch. Install project dependencies into a separate target directory.
Do not modify another project's environment, cancel another session's jobs, or use a login node for model inference.

Initial submission was rejected before allocation: `Invalid generic resource (gres) specification`.
Live `sinfo` shows `mi3501x` has no advertised GRES. Existing one-GPU jobs request a node without a GPU flag.
The repair removes the redundant GPU flag and keeps the same one-node partition, one-hour budget, model, and tests.
The failed submission's exact script and return code are preserved in `jobs.jsonl`.

Job 536316 loaded all model weights, then stopped on the native-logit control (maximum difference 0.12118).
Cause: the pilot softcapped BF16 head outputs. Native inference casts head outputs to FP32 before softcapping.
Repair: use the same FP32 softcap and recompute the final reader with the original full-sequence shape.
The model, prompts, weights, and interventions are unchanged. Preserve `state-scale-v1` and run the repair in `state-scale-v2`.
This is a measurement-code repair, not evidence about Inkling's behavior.

## Completed stages

| Job | Stage | Scheduler result | Elapsed |
|---|---|---|---|
| 536316 | First native-readout check | FAILED, 1:0 | 64 s |
| 536317 | Repaired forward and intervention pilot | COMPLETED, 0:0 | 39 s |
| 536326 | Matched saved-bank readouts | COMPLETED, 0:0 | 12 s |
| 536342 | Plain-completion capability checks | COMPLETED, 0:0 | 62 s |
| 536354 | Official-chat formatting diagnostic | COMPLETED, 0:0 | 79 s |

Total accepted allocation time was 256 seconds, with at most one owned GPU job running at once.
These terminal states were read with `sacct`; they are not claims about any other session's jobs.

The chat diagnostic initially used ID 200003 as an extra stop token. The checkpoint defines it as the tool-message header.
The actual text-message boundary is ID 200010. Saved raw streams remain unchanged.
The France stream includes a later channel after its completed text answer. Interpret only the first text block.
`src/chat_check.py` now derives both stop IDs from the tokenizer. No main pilot result depends on this correction.

All five run stages were uploaded using `push-run inkling-interp` to the configured private dataset.
The immutable archive revision and verified file inventories are in `evidence/artifact_archives.json`.
Model weights and the small gate-weight extract were omitted. Recorded inputs, logits, readouts, completions, and runtime logs were retained.

The human-facing `REPORT.md` and `DETAILS.md` are local drafts for user review.
The user requires results review before external reporting. Do not publish those drafts without that review.

## Continued research requested on 2026-11-21

The user requested further creative work, including repeated recent rotations.
The continuation kept the same pinned checkpoint and used sequential MI210 jobs.
It captured 144 calibration paragraphs and 55 development items, then fit transport maps.
The reserved 43-item comparison was frozen before its outputs were opened.
It did not confirm a mean/scale reader improvement.

Subsequent stages fit four country Jacobian rows, validated country exchanges,
compared intervention sites and matched edit sizes, and traced recovery after early edits.
Long-loop diagnostics included twelve development inputs through 32 passes and
an independent eight-paragraph confirmation through 64 passes.
Original model weights and all completed plans/configs were preserved.

Exact submissions, code commits, commands, terminal states, and exits are in `jobs.jsonl`.
The final scheduler snapshot is `evidence/scheduler_observation.json`.
Accepted continuation allocation time was 4562 seconds (76.03 GPU minutes), including failed and replacement stages.
At most one owned GPU job ran at once. No owned research job remains active.
No other session's jobs were stopped or changed.

The matched-size comparison initially failed because an equivalent unit-vector
formula changed BF16 rounding. The replacement reused the original calculation
and passed all twelve bitwise reference-logit checks. The failed partial directory
and its worker log were archived alongside the replacement.
See `plans/matched_country_direction_repair.md`.

All executed stages, including failures, have verified private immutable archives
in `evidence/artifact_archives.json`. A pending capture cancelled before allocation
had no model outputs; its scheduler receipt remains in the registry.

Later submissions run from detached worktrees at the exact source commit.
The earlier shared-checkout limitation is documented in `plans/source_provenance_note.md`.
The code backup remains `origin/inkling-interp`; never merge this independent history into Miles main.

The short overview is local `REPORT.md`, with three standalone experiment reports
under `reports/` and full methods in local `DETAILS.md`.
These drafts are stored on local branch `review/continued-research` and are not pushed.
The initial report remains on `review/initial-report`.
Keep human-facing drafts local until the user reviews them.


## Literature mapping requested on 2026-11-22

The user redirected the next step to a broad technique-by-technique map.
This stage performed literature reading and first-principles analysis only.
No model inference, new experiment, Slurm submission or job modification ran.
The inventory covers 86 entries in eight families and links 80 sources.
The source inventory and structural validation are in `evidence/interpretability_survey_*.json`.
Every assessment of looping is explicitly labeled as inference, not a measured ranking.

The local user-review drafts are `SURVEY.md` and
`reports/TRANSFORMER_TECHNIQUES_AND_LOOPING.md`, with a machine-readable catalog.
They are preserved on local branch `review/looping-survey` and are not pushed.
The source bibliography and operational state may be backed up normally.
All earlier experimental reports and review branches remain preserved.
