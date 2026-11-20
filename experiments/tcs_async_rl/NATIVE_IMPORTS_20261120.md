# Native training import diagnostic

The compatible vLLM image is generating the fixed-policy group in job 534098.
It lacks the SGLang, Megatron, Ray and Transformer Engine training stack.
The original native image failed GPU kernels on MI210. Its MI350 check remains queued as 534034.
No native training launcher has run yet.

Before allocating eight GPUs, run `python3 train_async.py --help` in the original native image.
Use the already cached pinned image and a fresh detached checkout and run directory.
`native_imports_v1.json` bounds this diagnostic to one MI210 for ten minutes.
The launcher imports its native dependencies and builds argument help. It starts no Ray cluster or training.
It reads no private question, calls no judge, and changes no scientific setting.
Keep all output and failures. A pass qualifies imports and argument availability only.
It does not qualify GPU kernels, model loading, backward computation, or MI350 compatibility.

This is an authorized deployment diagnostic under the user's instruction to debug the approved RL run.
The RL submission gate still requires a complete group with nonconstant real rubric rewards.
