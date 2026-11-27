# Skip the absent first-layer block-reader module

The first native-reader comparison stopped before inference. Physical layer0
has no attention_input module because no earlier layer outputs exist there.
The worker tried to install its generic reader hook on all24 layers. Skip
the absent module. None of the ten frozen read visits is physical layer0,
so the tested methods, read sites, data and metrics are unchanged.

The runtime source confirms attention_input is None only for layer_idx0.
The existing exact ten-readout guard validates the required modules at runtime.
Preserve the failed run and all original frozen inputs. The earlier scheduler
rejection allocated no GPU and is separately recorded in jobs.jsonl.

Rerun the unchanged frozen test as native-reader-lens-v2, one MI210 devel
allocation, at most20 minutes (1200 GPU seconds). Reserve the full limit and
count the failed allocation within the existing five-hour program. This is
an inference repair covered by the user's standing authorization.
