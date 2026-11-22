# Repair the initial weight check for a trained checkpoint

The second continuation loaded the actor checkpoint at iteration 24 and the original reference at iteration 0.
It failed before generation or optimization. Initialization compared the trained actor with the original Hugging Face weights.
That comparison cannot hold after training. One reported difference was 0.00000906 in a language attention parameter.

The actual restored publication matches the previous run after its 25th saved update.
All three engines agree on all 1,416 tensor hashes. None differs from the source publication.
This verifies the restored inference weights against saved evidence without loading another model or calling a judge.

The repair keeps equality checking enabled. A resumed driver snapshots the restored engine weights,
resets the checked tensors, rebroadcasts the same actor version, and compares with that snapshot.
Fresh starts retain the original Hugging Face comparison. Per-publication engine checksum logging remains enabled.
Four CPU protocol controls pass, including an incomplete rebroadcast that must fail.
The 95 TCS tests, syntax checks and formatting checks pass. The actual GPU round trip still needs qualification.

The approved Qwen3.5-4B setup, 20 questions, prompts, rewards, optimizer and sampling profile remain unchanged.
The extra startup reset consumes inference-server random draws before generation.
The configured seed remains 20261118. This is not a bitwise continuation of an earlier inference random stream.
FOCS/STOC 2026 remain untouched. Human calibration and independent-provider verification remain pending.

The failed jobs used 258 and 271 seconds. The replacement requests 3h51m and 40 new updates from the same checkpoint.
All attempts together can use at most 3h59m49s, or 31.976 GPU-hours. The judge cap remains $5.
The user authorized diagnosis, repair and replacement. Used configs and prior results remain preserved.
See [the exact evidence and updated run plan](evidence/initial_weight_resume_repair_20261122.json).
