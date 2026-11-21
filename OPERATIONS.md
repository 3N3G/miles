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
