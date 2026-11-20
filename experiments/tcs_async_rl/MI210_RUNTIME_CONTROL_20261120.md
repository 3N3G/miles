# Basic GPU operations with an existing project image as a software control

Primary-image probe 534037 enumerated MI210 but failed creating FP32 ones.
Bundled-library probe 534048 instead failed GPU initialization after six seconds.
The second result does not identify the cause of the first. It does not qualify bundled-library mode.

Use one ten-minute MI210 allocation for three fresh processes:

1. Current pinned SGLang image with the host-library wrapper.
2. Current pinned SGLang image with bundled userspace and explicit device bindings.
3. The existing vLLM image referenced by the user's AMD project, with the host-library wrapper.

Keep the device, kernel checks, private-data boundary, and wrapper settings fixed between the first and third cases.
The existing image is a software-control candidate. Prior use does not prove it works on this MI210 node.
Verify its recorded SHA-256 in the allocation before using it. Do not download or mutate it.
Read only GPU/runtime metadata and execute FP32 creation and BF16 matrix multiplication.
Save device-node permissions, driver/kernel versions, and loaded GPU-library paths before GPU initialization.
This preserves evidence even when device enumeration fails.

Continue to the control after an expected primary failure. Record each case as passed or failed.
Exit with failure if none passes. A passing case qualifies only these basic operations, not SGLang, Qwen, or training.
No model weights, dataset rows, API calls, or optimizer updates occur.
The source is `mi210_runtime_comparison_v1.json`, with a ten-GPU-minute cap including image verification.
MI350 group 534034 remains the valid queued generation fallback.
