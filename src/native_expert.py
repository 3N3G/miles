"""Exact routing-weight edits for one physical expert module, with visit audits."""

import torch


class RouterEdit:
    def __init__(self, expert=None, visits=(1, 2, 3), renormalize=False, observe=False,
                 position_mask=None):
        self.expert = expert
        self.visits = set(visits)
        self.renormalize = renormalize
        self.observe = observe
        self.position_mask = position_mask
        self.calls = 0
        self.routes = []
        self.audit = []

    def __call__(self, module, args, output):
        self.calls += 1
        logits, weights, indices = output
        if self.observe:
            self.routes.append((indices.detach().cpu().numpy(),
                                weights.detach().cpu().numpy()))
        if self.expert is None or self.calls not in self.visits:
            return output
        mask = indices == self.expert
        if self.position_mask is not None:
            mask = mask & self.position_mask.to(indices.device).reshape(-1, 1)
        removed = weights.masked_fill(~mask, 0).sum(-1)
        edited = weights.masked_fill(mask, 0)
        if self.renormalize:
            edited = edited * (weights.sum(-1, keepdim=True)
                               / edited.sum(-1, keepdim=True).clamp_min(1e-20))
        self.audit.append({"visit": self.calls, "selected_tokens": int(mask.any(-1).sum()),
                           "total_tokens": len(indices), "removed_mass": float(removed.sum()),
                           "max_removed_mass": float(removed.max()),
                           "renormalized": self.renormalize})
        return logits, edited, indices


def aliases(tokenizer):
    groups = {x: [] for x in "ABCD"}
    for i in range(len(tokenizer)):
        text = tokenizer.decode([i]).strip()
        if text in groups:
            groups[text].append(i)
    assert all(groups.values())
    assert len(set(sum(groups.values(), []))) == sum(map(len, groups.values()))
    return groups


def letter_metrics(logits, groups, answer):
    logp = logits.float().log_softmax(-1)
    scores = torch.stack([torch.logsumexp(logp[groups[x]], -1) for x in "ABCD"])
    probabilities = scores.softmax(-1)
    gold = "ABCD".index(answer)
    return {"gold_log_probability": float(scores[gold]),
            "letter_mass": float(scores.exp().sum()),
            "choice_probabilities": probabilities.tolist(),
            "prediction": "ABCD"[int(scores.argmax())],
            "correct": int(scores.argmax()) == gold,
            "gold_choice_log_probability": float(scores.log_softmax(-1)[gold])}


def render_question(record, permutation=None):
    permutation = list(range(4)) if permutation is None else permutation
    text = "Answer the following multiple-choice question. Reply with only the letter A, B, C, or D.\n\n"
    text += record["question"] + "\n"
    text += "\n".join(f"{'ABCD'[i]}. {record['choices'][old]}" for i, old in enumerate(permutation))
    text += "\nAnswer:"
    answer = "ABCD"[permutation.index(record["answer"])]
    return text, answer


def formatted_prompt(tokenizer, text, fmt):
    if fmt == "plain":
        return text
    assert fmt == "chat_text"
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False) + "<|message_model|><|content_text|>"
