# Complete the screen after a malformed judge verdict

Screen548472 stopped after157 completed evaluations and1012 allocation seconds.
The judge reported stop but returned only two of five required decisions.
The strict parser rejected the response. No provider failure became a zero reward.
The screen made zero optimizer updates and reached no medium solver answer.

The repair retries only structurally invalid verdicts, at most three attempts per unchanged request.
Every paid raw attempt remains preserved. Valid verdicts are never retried.
Cached attempts survive process restart without another charge or overwritten evidence.
The exact failed request passed on its next attempt with five decisions and score6/7.
The original malformed attempt remains byte-identical in both the old run and the repair cache.
The model, effort, prompt, rubric weights and paid-judge cap remain fixed.

The replacement uses checkpoint_screen_high_v3.json and a new output directory.
It repeats the same twelve questions, seed, checkpoint108 and32 samples per question.
The failed partial screen remains diagnostic and is not pooled into the complete baseline.
Valid cached verdicts can be reused for identical candidate text and request fingerprints.
The prior732 settled calls total$0.7305092, with no outstanding reservations.
Only the replacement owns the transferred ledger. The prior service was stopped first.

The new allocation is95 minutes on eight MI350 GPUs.
Prior screen allocations used272+1012=1284 seconds.
The combined maximum is(1284+5700)*8/3600=15.52 GPU-hours, below the approved16.
The twelve-hour RL allocation is separate and has not started.

Validation:144 TCS CPU checks cover retry bounds, invalid-schema rejection,
valid-score cache reuse, crash recovery without rebilling, and existing runtime contracts.
The actual failed provider response also passed the repaired request path.
The private failure archive and new receipt are recorded in the job registry.
