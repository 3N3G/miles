"""Descriptive native routing inspection on fit and development tokens only."""

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

NUMERIC_WORDS = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion first second third fourth fifth sixth seventh eighth ninth tenth day days week weeks month months year years hour hours minute minutes century centuries decade decades".split())


def numeric_cue(text):
    return any(c.isdigit() for c in text) or text.strip().casefold() in NUMERIC_WORDS


def plot_routing(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [r for r in result["cue_enrichment"] if r["split"] == "validation" and r["expert"] == 105]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(3)
    for offset, field, total, color, label in [(-.18, "selected_with_cue", "tokens_with_cue", "#0072B2", "Numeric or time-unit cue"),
                                               (.18, "selected_without_cue", "tokens_without_cue", "#999999", "Other token")]:
        values = [100*r[field]/r[total] for r in selected]
        ax.bar(x+offset, values, width=.34, color=color, label=label)
        for position, value, row in zip(x+offset, values, selected):
            ax.text(position, value+1.5, f"{row[field]}/{row[total]}", ha="center", fontsize=9)
    ax.set(xticks=x, xticklabels=[1, 2, 3], xlabel="Loop pass", ylabel="Tokens routed to expert 105 (%)", ylim=(0,100),
           title="Routing to one native expert across loop passes")
    ax.legend(loc="upper right")
    fig.subplots_adjust(bottom=.23)
    fig.text(.12, .035, "Post hoc analysis of 16 development articles. Counts label the bars.\nCue = digit, number word, ordinal or time-unit word. Selection is not causal necessity.", fontsize=8)
    fig.savefig(out/"native_expert_numeric_cues.png", dpi=180)
    fig.savefig(out/"native_expert_numeric_cues.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    records = json.loads((args.run/"inputs.json").read_text())["records"]
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    result = {"status": "Post hoc descriptive routing analysis, not a new semantic benchmark",
              "selected_experts": [105, 45, 3],
              "selection": "Native decoder directions aligned with the previously selected surrogate features",
              "cue_definition": "Current decoded token contains a digit or exactly matches a listed number, ordinal or time-unit word",
              "numeric_words": sorted(NUMERIC_WORDS), "final_article_states_read": False,
              "limits": ["Visible lexical cues are not abstract number representations.", "Development examples were already observed.", "Routing selection does not prove causal necessity or a unique expert role."],
              "contexts": [], "cue_enrichment": [], "expert105_bootstrap": []}
    for split in ["fit", "validation"]:
        selected = [(i, r) for i, r in enumerate(records) if r["split"] == split]
        all_indices, all_weights, cues = [], [], []
        for i, record in selected:
            data = np.load(args.run/f"routing_{i:03}.npz")
            all_indices.append(data["indices"])
            all_weights.append(data["weights"])
            cues.append([numeric_cue(tokenizer.decode([token])) for token in record["token_ids"][16:64]])
        indices, weights, cue = np.stack(all_indices), np.stack(all_weights), np.array(cues)
        if split == "validation":
            provenance = json.loads((args.run/"article_provenance.json").read_text())
            assert provenance["validation_unique_articles"] == len(selected) == 16
            rng = np.random.default_rng(1729)
            sampled = rng.integers(0, 16, size=(10000, 16))
            ncue, notcue = cue.sum(-1), (~cue).sum(-1)
            for p in range(3):
                hit = (indices[:, :, p] == 105).any(-1)
                yes, no = (hit & cue).sum(-1), (hit & ~cue).sum(-1)
                rate_a = yes[sampled].sum(-1)/ncue[sampled].sum(-1)
                rate_b = no[sampled].sum(-1)/notcue[sampled].sum(-1)
                ratios = rate_a/rate_b
                bounds = np.quantile(ratios, [0.05/(2*3), 1-0.05/(2*3)])
                result["expert105_bootstrap"].append({"pass":p+1,"rate_ratio_interval":bounds.tolist(),
                      "resamples":10000,"seed":1729,"sampling_unit":"development article",
                      "adjustment":"Bonferroni across three pass intervals, nominal family coverage 95 percent",
                      "limit":"Post hoc expert and cue choices are not accounted for by this interval."})
        for expert in result["selected_experts"]:
            for p in range(3):
                score = (weights[:, :, p] * (indices[:, :, p] == expert)).sum(-1)
                order = np.argsort(score.reshape(-1))[::-1][:4]
                contexts = []
                for flat in order:
                    document, position = int(flat//48), int(flat%48)+16
                    record = selected[document][1]
                    contexts.append({"id": record["id"], "position": position,
                                     "weight": float(score[document, position-16]),
                                     "token": tokenizer.decode([record["token_ids"][position]]),
                                     "context": tokenizer.decode(record["token_ids"][max(0, position-8):position+1])})
                result["contexts"].append({"split": split, "expert": expert, "pass": p+1,
                                            "selected_positions": int((score>0).sum()),
                                            "total_positions": score.size, "top_contexts": contexts})
        for expert in range(128):
            for p in range(3):
                hit = (indices[:, :, p] == expert).any(-1)
                a, b = int((hit & cue).sum()), int((hit & ~cue).sum())
                na, nb = int(cue.sum()), int((~cue).sum())
                pa, pb = a/na, b/nb
                result["cue_enrichment"].append({"split": split, "expert": expert, "pass": p+1,
                      "selected_with_cue": a, "tokens_with_cue": na,
                      "selected_without_cue": b, "tokens_without_cue": nb,
                      "selection_rate_with_cue": pa, "selection_rate_without_cue": pb,
                      "rate_ratio": pa/pb if pb else None})
    (args.run/"native_expert_contexts.json").write_text(json.dumps(result, indent=2)+"\n")
    if args.plot:
        plot_routing(result, args.run)
    for row in result["cue_enrichment"]:
        if row["expert"] == 105:
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
