"""Check the capability and unspoken-variable assumptions on the frozen prompts."""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from pilot import MODEL, REVISION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map={"": "cuda:0"}).eval()
    items = json.loads(args.items.read_text())["items"]
    items = [r for r in items if not r["name"].startswith("control-") or r["name"] == "control-spider"]
    rows = []
    with torch.inference_mode():
        for item in items:
            prefix = tokenizer(item["prompt"], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            ids, generated = prefix, []
            for step in range(12):
                output = model(input_ids=ids, use_cache=False).logits[0, -1].float()
                token = int(output.argmax())
                if step == 0:
                    base = json.loads((args.run / f"{item['name']}.json").read_text())["final_tokens"][0]["id"]
                    assert token == base, (item["name"], token, base)
                generated.append(token)
                if token == 200006:
                    break
                ids = torch.cat([ids, ids.new_tensor([[token]])], dim=-1)
            text = tokenizer.decode(generated)
            leading = bool(re.match(r"^[\s\"']*" + re.escape(item["target"]) + r"(?!\w)", text, flags=re.IGNORECASE))
            spoken = [c for c in item["intermediates"] if re.search(r"(?<!\w)" + re.escape(c) + r"(?!\w)", text, flags=re.IGNORECASE)]
            row = {"name": item["name"], "prompt": item["prompt"], "frozen_target": item["target"], "intermediates": item["intermediates"], "generated_ids": generated, "generated_text": text, "leading_target_match": leading, "bridge_mentioned_in_12_tokens": spoken}
            rows.append(row)
            print(json.dumps(row), flush=True)
            args.out.write_text(json.dumps({"model": MODEL, "revision": REVISION, "render": "verbatim plain completion", "decoding": "greedy argmax, at most 12 new tokens, use_cache=False", "scope": "Behavioral diagnostic only. Frozen targets are not assumed factually valid. Exact leading-word matches require manual context checks.", "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
