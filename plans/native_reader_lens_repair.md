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

## Correct censoring for the bounded generation control

The unchanged run's first arithmetic prompts begin with reasoning text and
reach the12-token budget. Such prefixes do not establish inability to answer
or that the intermediate stays unspoken. Before scoring the completed test,
make capability and unspoken status unknown for truncated generations. The
pinned generation_config.json has eos_token_id200006. Require this terminal
token for those secondary gates. Report truncated counts explicitly.
This changes no native inference, primary token-recovery endpoint or readout.
No answer-capability or hidden-reasoning claim is made from incomplete text.
