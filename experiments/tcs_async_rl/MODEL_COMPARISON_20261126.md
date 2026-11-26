# Qwen3.5-4B and Inkling 2 Mini

Inkling 2 Mini is a plausible stronger model, but we have no matched TCS proof comparison.
Our current policy is Qwen3.5-4B with additional easy and mixed-question reinforcement learning.
Published base-model benchmarks do not measure that trained policy.

| Property | Qwen3.5-4B | Inkling 2 Mini |
|---|---:|---:|
| Total parameter count | About4 billion |26.4 billion |
| Parameters selected per token | About4 billion |3.4 billion |
| Weight applications including repeated layers | About4 billion |5.4 billion |
| Approximate raw BF16 weight bytes |8GB |52.8GB |
| Publisher-reported LongBench v2 score |50.0% |66.9% |

Inkling uses a mixture of experts and runs its middle layers three times per token.
The advertised approximately20GB footprint uses low-precision inference. It is not a BF16 training-memory estimate.
The two LongBench scores come from separate publisher evaluations. They do not establish a controlled performance gap on proofs.
Sources: [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-4B) and
[Inkling release](https://thinkingmachines.ai/news/inkling-2-mini/).
Weight memory above is calculated as two bytes per parameter and excludes every other allocation.

For our experiment, Qwen already has validated AMD inference, checkpoint conversion and training in this fork.
Inkling 2 Mini would require checking its looped architecture in both training and inference, plus weight transfer.
No such AMD integration test has run. Existing older Inkling support does not establish Inkling 2 compatibility.

Tinker offers a free Inkling 2 Mini option with64K training and sampling limits, subject to throttling and account requirements.
The paid256K training option supports our90K responses. It uses adapter training rather than this local full-parameter pipeline.
See [current Tinker model limits and prices](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/).
Switching models or training systems would be a new experiment. No switch has been made.
