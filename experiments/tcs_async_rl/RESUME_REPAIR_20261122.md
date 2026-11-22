# Repair continuation loading and metric history

The first continuation failed before any new optimizer update.
The checkpoint scheduler stored a 40-update horizon. The continuation requested 65 total updates.
Loading compared the corresponding counters, 1,280 and 2,080, and stopped on their mismatch.

The approved setup remains Qwen3.5-4B with thinking, the same 20 easy TCS questions,
32 answers per update, 96 concurrent answers, CISPO and a constant learning rate of 1e-6.
The starting checkpoint retains 25 updates. The replacement targets 40 more within four hours on eight MI350 GPUs.
Luna medium keeps the same binary rubric and two anonymous orders, with a separate $5 ledger.
FOCS/STOC 2026 remain untouched. An example training question asks for a lower bound on matching-algorithm sensitivity.

The repair enables `use_checkpoint_opt_param_scheduler` to restore the saved scheduler state.
It preserves optimizer moments and the saved position. No training hyperparameter, prompt, data or reward changes.
A CPU control uses the installed scheduler's actual source, replacing only its logging import.
It reproduces the mismatch with the old flag. The repaired flag restores counter 800 and reaches 2,080 after 40 steps.
Every observed learning rate is 1e-6. Weight decay remains 0.1.

Checkpoint restoration also copies historical metric events into the new directory.
The reader now excludes events before the approved starting iteration.
The failed attempt's 25 inherited W&B points remain preserved and explicitly labeled as zero new training updates.
Future VM-native histories use a distinct online run ID.

Ninety-five TCS tests pass. Native GPU resume, entropy and update-direction observations still need qualification.
The replacement is an authorized runtime repair. Its used configuration will remain unchanged.
Detailed sources, counters and the failed run's archive are in [the operational record](evidence/resume_repair_20261122.json).

Slurm rounded the requested replacement limit of 3h55m42s upward to 3h56m.
The owned pending job was reduced to 3h55m and verified through `scontrol`.
Both attempts together can therefore use at most 3h59m18s, or 31.907 GPU-hours.
The renderer now floors limits to whole minutes. A regression test covers the rounding failure.
The submitted configuration and training source remain unchanged.
