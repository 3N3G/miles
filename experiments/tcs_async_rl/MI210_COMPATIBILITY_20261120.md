# MI210 kernel-image diagnostic

The v5 startup on MI210 job 534033 passed the cache-related imports, then failed at the first tensor operation.
The exact error was `hipErrorInvalidImage` / `device kernel image is invalid`.
No model weights, Qwen response, or reward measurement were reached.
The image reports Torch 2.11.0 with ROCm 10.0.0. GPU/driver compatibility remains unqualified.

This five-minute, one-GPU diagnostic separates two possible causes before another generation attempt:

1. In a fresh process, save the real GPU name, architecture, memory, and Torch's compiled architecture list.
   Test FP32 tensor creation and BF16 tensor creation/matrix multiplication without SGLang or AITER imports.
2. Only if that passes, repeat in a second fresh process after importing the engine.

The first failure localizes the problem to the base runtime or driver.
A failure only after engine import instead implicates imported extensions or their device state.
These results do not measure the TCS model or judge. Do not call missing generation a zero reward.

Use the same pinned image and cache repairs. Do not spoof GPU architecture, mutate the image, or change any model/data settings.
Keep repaired MI350 inference job 534034 queued. This failure has not been reproduced on MI350.
Record each diagnostic step before launching it, preserve failures, and upload the terminal artifacts.
The allocation cap is five GPU-minutes. There are no model downloads, private data reads, API calls, or optimizer updates.
