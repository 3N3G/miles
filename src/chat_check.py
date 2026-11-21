"""Diagnose whether plain-completion failures persist with the official chat format."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from pilot import MODEL, REVISION, baseline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    stop_ids = {tokenizer.convert_tokens_to_ids(t) for t in ["<|content_model_end_sampling|>", "<|end_message|>"]}
    model = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16, device_map={"": "cuda:0"}).eval()
    selected = {"control-france", "control-spider", "chem-photosynthesis-Z", "dbl-armistice-antonym", "firstletter-populous-country", "firstletter-greatwall-country"}
    rows = []
    with torch.inference_mode():
        for item in json.loads(args.items.read_text())["items"]:
            if item["name"] not in selected:
                continue
            rendered = tokenizer.apply_chat_template([{"role": "user", "content": item["prompt"]}], tokenize=False)
            rendered += "<|message_model|><|content_text|>"
            ids = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            logits, banks, gate_inputs, read = baseline(model, model.model.language_model, tokenizer, ids)
            generated = []
            for step in range(32):
                if step:
                    logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
                token = int(logits.argmax())
                generated.append(token)
                if token in stop_ids:
                    break
                ids = torch.cat([ids, ids.new_tensor([[token]])], dim=-1)
            row = {"name": item["name"], "prompt": item["prompt"], "rendered_prompt": rendered, "format": "official chat template, default effort 0.9, then explicit assistant text-channel prefix", "generated_ids": generated, "generated_text": tokenizer.decode(generated), "readouts": read}
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "readouts"}), flush=True)
            args.out.write_text(json.dumps({"model": MODEL, "revision": REVISION, "purpose": "Formatting diagnostic. Different conditioning from the plain pilot. No replacement of the frozen pilot scores.", "decoding": "greedy, at most 32 new tokens, no cache", "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
