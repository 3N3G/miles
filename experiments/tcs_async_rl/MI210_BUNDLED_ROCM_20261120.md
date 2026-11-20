# Test the bundled ROCm runtime on MI210

Compatibility job 534037 failed before importing SGLang or AITER.
Creating a 16 by 16 FP32 tensor raised `hipErrorInvalidImage`.
The actual device is MI210, `gfx90a:sramecc+:xnack-`, with 68,702,699,520 bytes.
Torch reports 2.11.0+rocm10.0.0 and includes `gfx90a` in its compiled architecture list.
This rules out the TCS model, rubric, and engine import as necessary triggers for this failure.
It does not establish which runtime library or driver component is incompatible.

## Hypothesis and bounded test

Apptainer's `--rocm` option injects host ROCm libraries.
The host configuration includes `libhsa-runtime64.so`, while this image ships a newer ROCm SDK.
A mixed runtime is one possible cause, not a confirmed diagnosis.
See [Apptainer GPU support](https://apptainer.org/docs/user/latest/gpu.html#amd-gpus-rocm).

Run the same five-minute diagnostic with the image's bundled userspace libraries.
Omit host library injection and explicitly bind `/dev/kfd` and `/dev/dri` from the allocated node.
This follows the device-passthrough approach described in [AMD's container documentation](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html).
Preserve Slurm/device visibility, the image digest, cache fixes, and all test operations.
Use a separate compilation-cache namespace for this runtime variant.
Save the loaded GPU-library paths, kernel release, architecture metadata, and the exact first failed operation.

Only if Torch-only FP32/BF16 checks pass, repeat after engine import in a second process.
A pass would qualify these basic operations for this device/runtime combination.
A failure would preserve evidence against this library-selection repair and require a different runtime or GPU.
Neither result measures Qwen answer quality or reward variance.

The snapshot is `mi210_compatibility_v2.json`. The cap remains five GPU-minutes.
No model weights, private data, provider credentials, judge calls, or optimizer updates are involved.
Repaired MI350 generation job 534034 remains queued because this GPU-kernel failure has not been reproduced there.
