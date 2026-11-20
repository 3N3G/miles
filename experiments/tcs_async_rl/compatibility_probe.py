"""Distinguish a Torch/driver failure from a failure introduced by engine imports."""

import argparse
import json
from pathlib import Path


def main():
    # GPU imports and operations are allowed only in the configured Slurm job.
    import os

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU compatibility checks require Slurm")
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("torch-only", "after-engine-import"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    properties = torch.cuda.get_device_properties(0)
    libraries = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if any(name in line for name in ("libhsa-runtime64", "libamdhip64", "libamd_comgr", "libdrm_amdgpu"))
        }
    )
    record = {
        "phase": args.phase,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": properties.name,
        "architecture": properties.gcnArchName,
        "bytes": properties.total_memory,
        "torch_arch_list": torch.cuda.get_arch_list(),
        "mapped_gpu_libraries": libraries,
        "kernel_release": os.uname().release,
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "step": "metadata",
    }
    output = args.run_dir / (args.phase + ".json")

    def mark(step):
        record["step"] = step
        output.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)

    try:
        mark("before-engine-import" if args.phase == "after-engine-import" else "before-fp32-ones")
        if args.phase == "after-engine-import":
            from sglang.srt.entrypoints.engine import Engine  # noqa: F401

        mark("fp32-ones")
        tensor = torch.ones((16, 16), device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()
        assert torch.equal(tensor.cpu(), torch.ones((16, 16)))
        mark("bf16-ones-and-matmul")
        tensor = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
        product = tensor @ tensor
        torch.cuda.synchronize()
        assert bool((product.cpu() == 16).all())
        mark("passed")
    except Exception as error:
        record.update(error_type=type(error).__name__, error=str(error))
        output.write_text(json.dumps(record, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
