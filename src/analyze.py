"""Reproduce the descriptive pilot analysis and smooth-gate algebra checks on CPU."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(0)
    samples = rng.choice(values, size=(10000, len(values)), replace=True).mean(-1)
    return {"mean": float(values.mean()), "bootstrap_95": np.quantile(samples, [0.025, 0.975]).tolist(), "n_prompts": len(values)}


def gate_analysis(run, items):
    weights = np.load(run / "gate_weights.npz")
    down, gain_matrix, forget_matrix = [weights[k].astype(np.float64) for k in ["down", "gain", "forget"]]
    rows, gains, keeps = [], [], []
    for item in items:
        name = item["item"]["name"]
        inputs = np.load(run / f"{name}.npz")["gate_inputs"].astype(np.float64)
        for pass_idx, (x, m) in enumerate(inputs, 2):
            norm = lambda z: z / np.sqrt(np.mean(z * z) + 1e-6)
            state = down @ np.r_[norm(x), norm(m)]
            gain, forget = np.tanh(gain_matrix @ state), np.tanh(forget_matrix @ state)
            gains.append(gain)
            keeps.append(1 - forget)
            r = np.sqrt(np.mean(m * m) + 1e-6)
            message_down = down[:, len(m) :]
            control_jacobian = message_down / r - np.outer(message_down @ m, m) / (len(m) * r**3)
            for seed in [0, 1, 2]:
                v = np.random.default_rng(seed).normal(size=len(m))
                coefficient = np.linalg.lstsq(control_jacobian.T, v, rcond=None)[0]
                v -= control_jacobian.T @ coefficient
                v *= np.linalg.norm(m) / np.linalg.norm(v)

                def forward(message):
                    u = down @ np.r_[norm(x), norm(message)]
                    return (1 - np.tanh(forget_matrix @ u)) * x + np.tanh(gain_matrix @ u) * message

                step = 1e-5
                numerical = (forward(m + step * v) - forward(m - step * v)) / (2 * step)
                predicted = gain * v
                row = {"name": name, "pass": pass_idx, "seed": seed, "control_rank": int(np.linalg.matrix_rank(control_jacobian)), "nonzero_gain_entries": int(np.count_nonzero(gain)), "rank_lower_bound": int(np.count_nonzero(gain)) - 64, "control_slope_per_unit": float(np.linalg.norm(control_jacobian @ v) / np.linalg.norm(v)), "transmission_ratio": float(np.linalg.norm(predicted) / np.linalg.norm(v)), "finite_difference_relative_error": float(np.linalg.norm(numerical - predicted) / np.linalg.norm(predicted)), "negative_gain_fraction": float(np.mean(gain < 0)), "keep_above_one_fraction": float(np.mean(1 - forget > 1))}
                rows.append(row)
    assert max(r["control_slope_per_unit"] for r in rows) < 1e-10
    assert max(r["finite_difference_relative_error"] for r in rows) < 1e-5
    summary = {"site": "stored layer 6, feed-forward message gate, final prompt token", "prompts": len(items), "pass_inputs": 2 * len(items), "nullspace_directions": len(rows), "calculation": "FP64 smooth re-evaluation using released gate weights and recorded forward inputs. This checks real-valued gate algebra, not BF16 rounding or semantic capacity.", "min_rank_lower_bound": min(r["rank_lower_bound"] for r in rows), "min_control_rank": min(r["control_rank"] for r in rows), "max_control_slope_per_unit": max(r["control_slope_per_unit"] for r in rows), "transmission_ratio_median": float(np.median([r["transmission_ratio"] for r in rows])), "transmission_ratio_range": [min(r["transmission_ratio"] for r in rows), max(r["transmission_ratio"] for r in rows)], "max_finite_difference_relative_error": max(r["finite_difference_relative_error"] for r in rows), "gain_median": float(np.median(gains)), "keep_median": float(np.median(keeps))}
    return summary, rows, np.array(gains), np.array(keeps)


def intervention_analysis(items, figure_dir):
    controls = [r for r in items if r["interventions"]]
    result = {}
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True)
    for bank, (ax, color) in enumerate(zip(axes, ["#0072B2", "#D55E00"])):
        scale, angle = [], []
        for j, row in enumerate(controls):
            a = next(x["kl_base_to_changed"] for x in row["interventions"] if x["kind"] == "scale" and x["bank"] == bank and x["value"] == 2)
            b = np.mean([x["kl_base_to_changed"] for x in row["interventions"] if x["kind"] == "angle" and x["bank"] == bank])
            scale.append(a)
            angle.append(b)
            offset = (j - 1.5) * 0.045
            ax.plot([offset, 1 + offset], [a, b], color="0.75", linewidth=1, zorder=1)
            ax.scatter([offset, 1 + offset], [a, b], color=color, s=28, zorder=2)
        result[f"bank_{bank}"] = {"scale_2": interval(scale), "rotation_60_degrees": interval(angle), "rotation_to_scale_mean_ratio": float(np.mean(angle) / np.mean(scale))}
        ax.set(title=f"Saved block {bank + 1}", xticks=[0, 1], xticklabels=["Double magnitude", "Rotate 60°"], yscale="log", xlim=(-0.25, 1.25))
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Output KL divergence (nats, log scale)")
    fig.suptitle("Direction changes affect predictions more than rescaling")
    fig.text(0.5, 0.01, "Four prompts. Each rotation dot averages three seeds. Lines pair the same prompt. No error bars.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(figure_dir / "scale_vs_direction.png", dpi=180)
    fig.savefig(figure_dir / "scale_vs_direction.pdf")
    plt.close(fig)
    scale_rows = [x for r in controls for x in r["interventions"] if x["kind"] == "scale" and x["value"] != 1]
    result["rescaling_top1_agreement"] = {"same": sum(x["top1_agrees"] for x in scale_rows), "total": len(scale_rows)}
    result["sham_max_logit_change"] = max(x["max_logit_change"] for r in controls for x in r["interventions"] if x["kind"] == "scale" and x["value"] == 1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, default=Path("evidence"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.figures.mkdir(parents=True, exist_ok=True)
    documents = [json.loads(p.read_text()) for p in sorted(args.run.glob("*.json"))]
    items = [r for r in documents if "item" in r]
    assert len(items) == 16
    gate, rows, gains, keeps = gate_analysis(args.run, items)
    intervention = intervention_analysis(items, args.figures)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for values, ax, label, limits in [(gains, axes[0], "Message multiplier", (-1, 1)), (keeps, axes[1], "Current-input multiplier", (0, 2))]:
        for offset, color in [(0, "#CC79A7"), (1, "#009E73")]:
            v = np.sort(values[offset::2].ravel())
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, label=f"Pass {offset + 2}")
        ax.set(xlabel=label, ylabel="Fraction of coordinates", xlim=limits, ylim=(0, 1))
        ax.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle("The inspected gate usually transmits the message with large gain")
    fig.text(0.5, 0.01, "Layer 6 feed-forward message gate only. Smooth calculation at 32 recorded inputs. No error bars.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(args.figures / "gate_multipliers.png", dpi=180)
    fig.savefig(args.figures / "gate_multipliers.pdf")
    plt.close(fig)
    readouts = json.loads((args.run / "readout_comparison.json").read_text())
    comparison = []
    for method in ["raw", "centered", "delta", "centered_delta"]:
        selected = [r for r in readouts["rows"] if r["method"] == method and not r["filtered"] and r["loop"] >= 2 and not r["name"].startswith("control-")]
        names = sorted({r["name"] for r in selected if r["target_ids"]})
        hits = [name for name in names if any(r["hit"] for r in selected if r["name"] == name)]
        comparison.append({"method": method, "eligible": len(names), "hits": len(hits), "hit_names": hits})
    behavior = json.loads((args.run / "completions.json").read_text())["rows"]
    solved = {r["name"] for r in behavior if r["leading_target_match"] and not r["name"].startswith("control-")}
    unspoken = {r["name"] for r in behavior if r["name"] in solved and not r["bridge_mentioned_in_12_tokens"]}
    capability_subset = []
    for method in ["raw", "centered", "delta", "centered_delta"]:
        selected = [r for r in readouts["rows"] if r["method"] == method and not r["filtered"] and r["loop"] >= 2 and r["name"] in unspoken and r["target_ids"]]
        names = sorted({r["name"] for r in selected})
        hits = [name for name in names if any(r["hit"] for r in selected if r["name"] == name)]
        capability_subset.append({"method": method, "eligible": len(names), "hits": len(hits), "hit_names": hits})
    bank_states = np.array([np.load(args.run / f"{r['item']['name']}.npz")["banks"] for r in items])
    normalized = bank_states / np.sqrt(np.mean(bank_states**2, axis=-1, keepdims=True) + 1e-6)
    end_difference = normalized[:, 2, 1] - normalized[:, 1, 1]
    coordinate_share = end_difference[:, 2027] ** 2 / np.sum(end_difference**2, axis=-1)
    loop_geometry = {"known_coordinate": 2027, "status": "Post hoc replication of Goodfire's previously reported loop-bookkeeping coordinate, not a novelty claim.", "prompts": len(items), "coordinate_share_of_final_bank_transition_squared_norm_mean": float(coordinate_share.mean()), "coordinate_share_range": [float(coordinate_share.min()), float(coordinate_share.max())], "limitation": "Two transitions do not establish an asymptotic convergence trend. Normalization alone does not remove loop bookkeeping."}
    summary = {"native_logit_max_error": max(r["native_logit_error"] for r in items), "validated_prompts": len(items), "intervention": intervention, "gate": gate, "loop_geometry": loop_geometry, "readouts": comparison, "behavior": {"development_prompts": 12, "leading_frozen_target_matches": len(solved), "posthoc_correct_and_unspoken_subset": capability_subset, "limitation": "Post hoc diagnostic subset, not a preregistered or held-out score. Unspoken means absent from the first 12 generated tokens."}}
    (args.evidence / "pilot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.evidence / "gate_nullspace_checks.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
