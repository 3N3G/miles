"""CPU comparison of learned decoder directions with native expert neuron outputs."""

import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from operator_surrogate import SparseOperator

E2M1 = [0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.]


def decode_one(weight, scale, factor):
    lookup = torch.tensor(E2M1)
    low = lookup[(weight & 15).long()]
    high = lookup[(weight >> 4).long()]
    values = torch.stack([low, high], dim=-1).reshape(*scale.shape, 16)
    return (values * scale[..., None] * factor).reshape(weight.shape[0], -1).bfloat16()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    torch.set_num_threads(2)
    model = SparseOperator.from_checkpoint(torch.load(args.run/"shared_seed0.pt", weights_only=True), "cpu")
    features = json.loads((args.run/"feature_selection.json").read_text())["features"]
    dimensions = (128, 2304, 576)
    packed = np.memmap(args.weights/"w2_weight.raw", dtype=np.uint8, mode="r", shape=dimensions)
    scales = np.memmap(args.weights/"w2_weight_scale.raw", dtype=np.uint8, mode="r", shape=(128, 2304, 72))
    amax = torch.from_numpy(np.fromfile(args.weights/"w2_weight_amax.raw", dtype=np.float32)).bfloat16()
    denominator = torch.from_numpy(np.fromfile(args.weights/"w2_weight_outer_scale_denominator.raw", dtype=np.float32)).bfloat16()
    # Match the loader's source casts, then its inspected conversion operation.
    factor = amax.float() / denominator.float()
    source = args.official_source.read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Inkling2LoopDequantizeExperts")
    namespace = {"torch": torch, "ConversionOps": object}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(args.official_source), "exec"), namespace)
    converter = namespace["Inkling2LoopDequantizeExperts"]()
    patterns = ["weight", "scale", "amax", "denominator"]
    directions = model.decoder.weight[:, features].T.detach()
    generator = torch.Generator().manual_seed(1729)
    random = torch.nn.functional.normalize(torch.randn(len(features), 2304, generator=generator), dim=-1)
    directions = torch.cat([directions, random])
    directions = torch.nn.functional.normalize(directions, dim=-1)
    all_scores, conversion_errors = [], []
    with torch.inference_mode():
        for expert in range(128):
            weight = torch.from_numpy(np.array(packed[expert]))
            scale = torch.from_numpy(np.array(scales[expert])).view(torch.float8_e4m3fn).bfloat16().float()
            decoded = decode_one(weight, scale, factor[expert])
            if expert in [0, 31, 64, 127]:
                inputs = {"weight": weight[None].bfloat16(), "scale": scale[None].bfloat16(),
                          "amax": amax[expert:expert+1], "denominator": denominator}
                official = converter.convert(inputs, patterns, ["target"])["target"][0]
                error = (official.float()-decoded.float()).abs().max().item()
                assert error == 0, error
                conversion_errors.append({"expert": expert, "max_error": error})
            decoded = decoded.float()
            scores = (directions @ decoded) / decoded.norm(dim=0).clamp_min(1e-12)
            all_scores.append(scores)
    scores = torch.stack(all_scores, dim=1).reshape(2*len(features), -1)
    matches = []
    for index, feature in enumerate(features):
        values, indices = scores[index].abs().topk(5)
        matches.append({"feature": feature, "top_native_output_matches": [
            {"expert": flat//1152, "neuron": flat%1152, "cosine": float(scores[index, flat])}
            for flat in indices.tolist()]})
    result = {"status": "Post hoc weight alignment only, not proof of a recovered native computation",
              "native_units_searched": 128*1152,
              "conversion": "NVFP4 decoded to BF16 with the pinned Transformers formula and source casts",
              "official_conversion_checks": conversion_errors,
              "official_source_sha256": hashlib.sha256(args.official_source.read_bytes()).hexdigest(),
              "matches": matches,
              "random_direction_max_absolute_cosines": scores[len(features):].abs().max(-1).values.tolist(),
              "elapsed_cpu_wall_seconds": time.monotonic()-start,
              "limits": ["Only decoder output directions are compared.", "Native gating, encoder alignment and Jacobian agreement are not tested.", "No final article states or new native-model inference are used."]}
    (args.run/"native_decoder_alignment.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
