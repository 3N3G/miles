"""Recover the three small gate tensors using byte ranges, without downloading the model."""

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/ed40e7da625f594fc9c4990f5f66be24873c22d7/model-00001-of-00001.safetensors"


def read_range(start, length):
    request = urllib.request.Request(URL + f"?range_start={start}", headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-"):
            raise RuntimeError("Server did not honor the byte-range request")
        data = response.read(length)
        if len(data) != length:
            raise RuntimeError("Truncated tensor data")
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    header_length = int.from_bytes(read_range(0, 8), "little")
    header = json.loads(read_range(8, header_length))
    tensors = {}
    for name, suffix in [("down", "down_weight"), ("gain", "gain_weight"), ("forget", "forget_weight")]:
        spec = header[f"model.llm.layers.6.controllers.mlp.{suffix}"]
        start, end = spec["data_offsets"]
        raw = read_range(8 + header_length + start, end - start)
        if spec["dtype"] == "BF16":
            value = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        elif spec["dtype"] == "F32":
            value = np.frombuffer(raw, dtype="<f4")
        else:
            raise ValueError(spec["dtype"])
        tensors[name] = value.reshape(spec["shape"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **tensors)
    print(json.dumps({"out": str(args.out), "shapes": {k: list(v.shape) for k, v in tensors.items()}}))


if __name__ == "__main__":
    main()
