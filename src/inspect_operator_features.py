"""Post hoc feature inspection on development inputs. No native-model inference."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from operator_surrogate import SparseOperator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    inputs = json.loads((args.run / "inputs.json").read_text())["records"]
    validation = [i for i, r in enumerate(inputs) if r["split"] == "validation"]
    assert len(validation) == 16
    # Read only the development slice. The final articles are not revisited.
    data = np.load(args.run / "operator_inputs.npy", mmap_mode="r")
    x = torch.tensor(np.array(data[validation]))
    del data
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    model = SparseOperator.from_checkpoint(torch.load(args.run / "shared_seed0.pt", weights_only=True), "cpu")
    means = torch.load(args.run / "means.pt", weights_only=True)
    features = json.loads((args.run / "feature_selection.json").read_text())["features"]
    activations = []
    with torch.inference_mode():
        for p in range(3):
            activations.append(torch.cat([model.encode(batch) for batch in x[:, :, p].reshape(-1, 2304).split(128)]))
    rows = []
    for feature in features:
        examples, counts = [], []
        for p in range(3):
            z = activations[p][:, feature]
            counts.append(int((z > 0).sum()))
            values, indices = z.topk(4)
            contexts = []
            for value, flat in zip(values.tolist(), indices.tolist()):
                record = inputs[validation[flat // 48]]
                position = flat % 48 + 16
                ids = record["token_ids"]
                contexts.append({"id": record["id"], "position": position, "activation": value,
                                 "token": tokenizer.decode([ids[position]]),
                                 "context": tokenizer.decode(ids[max(0, position-8):position+1])})
            examples.append({"pass": p+1, "contexts": contexts})
        reuse = {}
        for a, b in [(0, 1), (0, 2), (1, 2)]:
            za, zb = activations[a][:, feature], activations[b][:, feature]
            intersection = int(((za > 0) & (zb > 0)).sum())
            union = int(((za > 0) | (zb > 0)).sum())
            reuse[f"{a+1}-{b+1}"] = {"active_position_intersection": intersection,
                                     "active_position_union": union,
                                     "jaccard": intersection/union if union else None,
                                     "activation_correlation": float(torch.corrcoef(torch.stack([za, zb]))[0, 1])}
        decoder = model.decoder.weight[:, feature]
        offsets = [torch.nn.functional.cosine_similarity(decoder, means[f"pass{p}"]-means["shared"], dim=0).item() for p in [1, 2, 3]]
        row = {"feature": feature, "active_development_positions_by_pass": counts,
               "positions_per_pass": 768, "reuse": reuse,
               "decoder_cosine_to_training_pass_mean_offsets": offsets,
               "examples": examples}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    result = {"status": "Exploratory development inspection, not a fresh semantic evaluation",
              "source": "Previously selected shared seed-zero features",
              "development_paragraphs": 16, "final_article_states_read": False, "features": rows}
    (args.run / "feature_development_inspection.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
