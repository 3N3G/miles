# Use a writable Megatron Bridge config-lock cache

Job562208 passed native argument validation and initialized all six rollout engines. Actor model initialization then failed because Megatron Bridge tried to create a configuration lock under the unmounted, read-only `/home1/geney` container home.
It used282 allocated seconds. The previous parser failure used37 seconds. Neither attempt produced dataset answers or optimizer updates.

The installed container source `megatron/bridge/models/hf_pretrained/safe_config_loader.py` explicitly supports `MEGATRON_CONFIG_LOCK_DIR`.
The repair sets that variable to the approved work directory's `cache/megatron-config-locks` and forwards it to every Ray worker.
The parent and workers use the same path. The home directory is not changed.
A real `AutoBridge.from_hf_pretrained` configuration preflight now runs before rollout-engine startup. It records success and the cache directory.

The public runtime environment filter has one owner. It selects permitted cache variables before reading their values. Credentials and HOME are excluded.
All35 CPU tests pass, including directory creation and child-environment forwarding. Targeted Ruff and whitespace checks pass.
GPU validation of the Bridge loader remains pending until the replacement allocation.

Preserve the scientific configuration unchanged. Start referencev4 from the same pinned original HF model.
Request6840 seconds, rounded down from6881 remaining seconds. Across attempts,319+6840=7159 seconds remains below the7200-second reference cap.
The250-update target,15-minute save reserve, fixed validation, original reward, CISPO gate and independent7200-second CISPO allowance remain unchanged.
The failed attempt's31 files are hash-verified at HF commit `d83867903016c9c62c755b05419b261883708bc0`, under `miles/gsm8k-reference-v3`.
The preceding diagnostic repair,3 GiB GSM8K monitor reserve and64 MiB portable-archive threshold continue to apply.
