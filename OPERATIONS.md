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


## Shared-operator pilot preparation on 2026-11-22

The user selected the shared-operator direction. The required allocation budget
has not yet been specified. No new GPU job or native-model inference ran.
Only job-list and existing-artifact inspections were performed on AMD.
The unrelated live job observed during that inspection was not modified.

The proposed experiment is `plans/shared_operator_pilot.md`. Its active config
is `evidence/operator_pilot_config.json`. The active fixed inputs are
`evidence/operator_pilot_inputs_articles.json`: 128 fit paragraphs, 16 development
paragraphs and one paragraph from each of 32 new final articles. The earlier
paragraph-sampling draft remains unchanged in `evidence/operator_pilot_inputs.json`.
No model outputs were collected from either draft during preparation.

Nine CPU checks passed, including capture/replay on a small CPU model and
invariance of fitting when the final values are replaced with extreme values.
The planted-teacher and shuffled-pair control results are implementation checks,
not evidence about Mini. Their numbers and environment are in
`evidence/operator_pilot_validation.json`.

Prepared submission, after the allocation decision is recorded and pushed:

```sh
.venv/bin/python src/submit_stage.py --name shared-operator-v1 \
  --plan plans/shared_operator_pilot.md --minutes 30 --partition mi2101x \
  python src/operator_pilot.py --config evidence/operator_pilot_config.json \
  --inputs evidence/operator_pilot_inputs_articles.json \
  --out /work1/aviralkumar/geney/inkling-interp-data/runs/shared-operator-v1
```

The program refuses model execution while the config's allocation status is
pending. After execution, produce plots with `src/analyze_operator_pilot.py`
on the VM using copied small analysis artifacts. Preserve all raw failures.
Archive results with `push-run` and record the private immutable path.
Keep trained surrogate checkpoints on the cluster, with hashes recorded, rather
than uploading model checkpoints. No mentor-facing reporting is authorized.


## Completed shared-operator pilot

The user approved the one-MI210, 30-minute cap with “yes” on 2026-11-22.
Job 541303 ran from detached source commit
`c6ea41d9f71801d2bdee4fe8883425896ca66fe9` and completed with exit 0:0.
`sacct` reports 00:05:50. The model program reports 337.91 seconds.
There were no replacement jobs and no other session's jobs were modified.
The submitted protocol is preserved in `plans/archive/shared_operator_pilot_submitted.md`.

All 15 fits, 54 final metric rows, 24 native replacements and 192 feature/random
edits completed. Replay and hook controls passed exactly. Original native logits
also remained unchanged after fitting. `src/validate_operator_pilot.py` checks
artifact counts, source/input hashes, split integrity, sham edits and norm matching.
Its numerical summary is `evidence/operator_pilot_result.json`.

Post hoc CPU work inspected development contexts, partial native-neuron output
alignment and numerical-token routing. No new native forward passes, fits or
GPU allocations were added. The original native decoder ranges were extracted
outside the run folder, using `src/extract_operator_decoder.py`, and are not
uploaded. Four conversion checks against the pinned class agree exactly.
The source and interpretation limits are in the analysis manifest and reports.

The original 16 development paragraphs map to 16 articles. Three also occur in
the fitting set. No final article overlaps either earlier set. Article mapping
is preserved in the private raw artifacts.

The final archive is `https://huggingface.co/datasets/3N3G/internalization-runs/tree/74d36b2f3eaf690fce37d6ea2455649103a696ad/inkling-interp/shared-operator-v1`.
The verification record confirms 208 entries, eight downloaded critical files
matching local hashes, private visibility, and no uploaded checkpoint or native
weight files. The earlier snapshot is also retained in the archive registry.
Training checkpoints remain on AMD, with 17 hashes in the archived manifest.

The short user-review report is local `reports/SHARED_OPERATOR.md`. Full methods
and raw examples are in `reports/SHARED_OPERATOR_DETAILS.md`. These drafts and
their builder are preserved on local branch `review/shared-operator` and are not
pushed. The report keeps the primary prediction test distinct from the post hoc
native expert finding. The next research choice is recorded in `QUESTIONS.md`.
## Autonomous native expert program, 2026-11-23

The user authorized autonomous expert research and at most five GPU hours.
`evidence/expert_program_budget.json` is the budget ledger. Refresh with
`python src/expert_budget.py`. Count every allocated GPU, including loading.
The earlier350-second pilot is counted conservatively. Submit one allocation
at a time. Queue time does not count. Never touch another session's jobs.

Primary text results and development findings are in
`evidence/native_expert_screen_result.json`. Both prospective numeric-loss
contrasts include zero despite replicated routing enrichment. Two failed
stages preserved valid earlier outputs. The derivative range and grouped-unit
observer were repaired with separate recorded plans and raw archives.

The independent MMLU confirmation specification is frozen in
`evidence/native_expert_confirmation_spec.json`. All16 conditions,128 questions,
three order seeds and12 corrected primary intervals must remain fixed.
The batched attempt failed on calibration before any test inference. Use only
`src/native_expert_confirmation.py` with the serial plan. Do not inspect
confirmation efficacy until all three fixed seeds complete. Progress logs,
completion manifests and inference-validity checks may be monitored.

Current and terminal job identities are in the ledger and `jobs.jsonl`.
Query the scheduler, rather than inferring live state from this note.
Remaining seed commands follow this template, replacing SEED with1 or2:

```
.venv/bin/python src/submit_stage.py --name expert-confirmation-sSEED --plan plans/native_expert_confirmation_serial.md --minutes 30 --partition devel python src/native_expert_confirmation.py --spec evidence/native_expert_confirmation_spec.json --inputs evidence/native_expert_inputs.json --out /work1/aviralkumar/geney/inkling-interp-data/runs/expert-confirmation-sSEED --seed SEED
```

A native unit-routing experiment is fully prepared. It may run between fixed
seeds without reading their efficacy or changing their protocol:

```
.venv/bin/python src/submit_stage.py --name native-unit-routing-v1 --plan plans/native_unit_block_routing.md --minutes 10 --partition devel python src/native_unit_routing.py --inputs evidence/unit_routing_inputs.json --out /work1/aviralkumar/geney/inkling-interp-data/runs/native-unit-routing-v1
```

Read the unit-routing plan before interpreting it. It tests an actual Block2
read weight, not a trained phase classifier or a global clock claim. Unit45:1127
changes sign on1113/1318 paired fitting positions. Unit3:664 also rises late.
Goodfire's existing block-tagging/outlier results are prior art. The new test
targets specific native units with exact grouped-backend counterfactuals.

After each job, record terminal state, update the budget and workstream, upload
with `src/archive_stage.py`, and rsync raw outputs to the external data folder.
For completed MMLU seeds, retain the blind until all three are complete. Then:

```
.venv/bin/python src/analyze_expert_confirmation.py --root /home/ubuntu/inkling-interp-data --spec evidence/native_expert_confirmation_spec.json --inputs evidence/native_expert_inputs.json --out /home/ubuntu/inkling-interp-data/expert-confirmation-analysis-v1
.venv/bin/python src/analyze_unit_routing.py /home/ubuntu/inkling-interp-data/native-unit-routing-v1
```

Human reports remain local review drafts under ignored `reports/`. They must
stand alone, state uncertainty, show raw examples and distinguish routing,
readout and intervention. Preserve them on a local review branch. Do not push
mentor-facing prose. Code/evidence backup is explicitly `HEAD:inkling-interp`.
