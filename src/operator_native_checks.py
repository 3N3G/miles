"""Native-model replacement and last-token feature edits after all fitting is frozen."""

import hashlib
import json

import numpy as np
import torch

from operator_surrogate import SparseOperator, VisitEdit
from pilot import token_list


@torch.no_grad()
def run(model, operator, ids, hook=None):
    handle = operator.register_forward_hook(hook) if hook else None
    try:
        logits = model(input_ids=ids, use_cache=False).logits.float()
    finally:
        if handle:
            handle.remove()
    if hook:
        assert hook.calls == 3, hook.calls
    return logits


@torch.no_grad()
def distribution_metrics(logits, reference, ids, tokenizer):
    p, q = reference.log_softmax(-1), logits.log_softmax(-1)
    kl = (p.exp() * (p - q)).sum(-1)
    labels = ids[:, 17:64]
    nll = -q[:, 16:63].gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    baseline_nll = -p[:, 16:63].gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return {"mean_kl_positions16_63": kl[:, 16:].mean().item(),
            "last_token_kl": kl[:, -1].mean().item(),
            "top1_agreement_positions16_63": (logits[:, 16:].argmax(-1) == reference[:, 16:].argmax(-1)).float().mean().item(),
            "next_token_nll_positions16_62": nll.mean().item(),
            "native_next_token_nll_positions16_62": baseline_nll.mean().item(),
            "native_top10": token_list(reference[0, -1], tokenizer),
            "edited_top10": token_list(logits[0, -1], tokenizer)}


@torch.no_grad()
def native_checks(model, tokenizer, records, shared, config, out):
    operator = model.model.language_model.layers[config["layer_index"]].mlp
    first_ids = torch.tensor([records[0]["token_ids"]], device="cuda")
    after_fitting = run(model, operator, first_ids)
    controls = json.loads((out / "native_controls.json").read_text())
    assert hashlib.sha256(after_fitting.cpu().numpy().tobytes()).hexdigest() == controls["first_input_logit_sha256"], "Native model changed during surrogate fitting"
    controls["native_logits_unchanged_after_fitting"] = True
    (out / "native_controls.json").write_text(json.dumps(controls, indent=2) + "\n")

    selected = [(i, r) for i, r in enumerate(records) if r["split"] == "validation"][:8]
    cached = np.load(out / "operator_inputs.npy", mmap_mode="r")
    last_inputs = torch.tensor(np.array(cached[[i for i, _ in selected], -1]), device="cuda")
    scores = shared.encode(last_inputs).sum(dim=(0, 1))
    count = min(4, int((scores > 0).sum()))
    features = scores.topk(count).indices.cpu().tolist()
    selection = {"rule": "Top four by summed last-token activation across eight fixed validation paragraphs and all three passes",
                 "features": features, "source_split": "validation", "final_data_used": False}
    (out / "feature_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    specifics = [SparseOperator.from_checkpoint(torch.load(out / f"pass{p+1}_seed0.pt", weights_only=True), "cuda") for p in range(3)]
    means = torch.load(out / "means.pt", weights_only=True)
    means = [means[f"pass{p+1}"].to("cuda") for p in range(3)]
    replacements, interventions = [], []
    for _, record in selected:
        ids = torch.tensor([record["token_ids"]], device="cuda")
        reference = run(model, operator, ids)
        identity = run(model, operator, ids, VisitEdit())
        assert torch.equal(reference, identity), "Identity hook changed logits"
        for name, replacement in [("shared", shared), ("pass_specific", specifics), ("pass_mean", means)]:
            logits = run(model, operator, ids, VisitEdit(replacement=replacement))
            replacements.append({"id": record["id"], "condition": name,
                                 "input": record["rendered_input"],
                                 **distribution_metrics(logits, reference, ids, tokenizer)})
        (out / "native_replacements.json").write_text(json.dumps(replacements, indent=2) + "\n")
        for feature in features:
            for p in range(3):
                for random in [False, True]:
                    hook = VisitEdit(replacement=shared, pass_index=p, feature=feature,
                                     random_seed=1729+feature if random else None)
                    logits = run(model, operator, ids, hook)
                    difference = logits[0, -1] - reference[0, -1]
                    indices = difference.abs().topk(10).indices.cpu().tolist()
                    row = {"id": record["id"], "feature": feature, "pass": p+1,
                           "direction": "matched_norm_random" if random else "feature",
                           "input": record["rendered_input"],
                           "intended_edit_norm": hook.edit_norms[0][0],
                           "actual_bf16_edit_norm": hook.actual_edit_norms[0][0],
                           "largest_logit_changes": [{"token_id": j, "token": tokenizer.decode([j]),
                                                      "change": float(difference[j])} for j in indices],
                           **distribution_metrics(logits, reference, ids, tokenizer)}
                    interventions.append(row)
        (out / "native_feature_edits.json").write_text(json.dumps(interventions, indent=2) + "\n")
        print(json.dumps({"event": "native_checks", "id": record["id"], "feature_edits": len(interventions)}), flush=True)
