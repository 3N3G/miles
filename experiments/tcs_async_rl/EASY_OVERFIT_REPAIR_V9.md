# Repair of the initial weight checker

Job 536321 stopped before generation or optimization. Native arguments, conversion and initial text-weight publication succeeded.
The checker confirmed equality for 331 text parameters and 81 derived normalization buffers on the logged rank.
All reported mismatches belonged to the frozen `visual.*` image encoder, which the text-only trainer does not publish.
The checker had randomized this unused branch while testing publication. No training answer or optimizer update ran.

The replacement snapshot is `rl_easy_overfit_v9.json`. It uses the existing skip-list flag to leave only `visual.*` untouched.
Keep strict checking for every text parameter and derived normalization buffer. Keep the same model, data, prompts,
reward, optimizer, sampling, topology, seed, 40-update limit and four-hour allocation.
The initial conversion is complete and can be reused. The submitted v8 snapshot and all results stay unchanged.

The VM tunnel repair disables SSH multiplexing for tracked node connections. The first detached connection was removed.
The allocated node then verified the exact 20-question service identity. Credentials stayed on the VM.
Judge controls and receipts carry forward as cached, question-specific results, along with their spending ledger.
No completed judge score is retried. The model-output and judge-token limits stay as documented in the approved plan.

Before replacement submission, 75 CPU tests, lint and whitespace checks passed.
The new native argument stage must confirm `--check-weight-update-skip-list visual.`.
The initial weight comparison must pass before any rollout is accepted.
Monitor the first real reward group, backward computation, weight publication and checkpoint saving.
This repair follows the user's authorization to submit and debug the RL run until it runs.

Evidence: [weight-check failure and repair](evidence/weight_check_repair_v9_20261121.json).
The experiment design remains [the approved easy-only plan](EASY_OVERFIT_PLAN_20261121.md).
