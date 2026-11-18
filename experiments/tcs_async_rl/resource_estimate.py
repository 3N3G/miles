"""Reproducible component estimates, not an allocation or a measured peak-memory report.

Run: python experiments/tcs_async_rl/resource_estimate.py --output /tmp/tcs-sizing.json
The proposed config owns the inputs. No model weights, credentials, or GPU APIs are loaded.
"""

import argparse
import json
import math
from pathlib import Path

GIB = 2**30


def model_inputs(config: dict, *, base_dir: Path) -> dict:
    """Derive model geometry and counts once from the pinned public source artifacts."""
    inputs = dict(config["sizing_inputs"])
    model = json.loads((base_dir / inputs.pop("model_config_file")).read_text())["text_config"]
    inventory = json.loads((base_dir / inputs.pop("parameter_inventory_file")).read_text())
    shapes = inventory["example_tensor_shapes"]
    inputs.update(
        text_parameters=inventory["parameter_counts"]["text"],
        mtp_parameters=inventory["parameter_counts"]["mtp"],
        vision_parameters=inventory["parameter_counts"]["vision"],
        tied_embedding_parameters=math.prod(shapes["model.language_model.embed_tokens.weight"]["shape"]),
        layers=model["num_hidden_layers"],
        full_attention_layers=model["layer_types"].count("full_attention"),
        attention_heads=model["num_attention_heads"],
        kv_heads=model["num_key_value_heads"],
        head_dim=model["head_dim"],
        hidden_size=model["hidden_size"],
        vocab_size=model["vocab_size"],
        linear_attention_layers=model["layer_types"].count("linear_attention"),
        linear_value_heads=model["linear_num_value_heads"],
        linear_key_dim=model["linear_key_head_dim"],
        linear_value_dim=model["linear_value_head_dim"],
        linear_conv_width=shapes["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"]["shape"][0],
        linear_conv_kernel=model["linear_conv_kernel_dim"],
    )
    return inputs


def kv_bytes_per_token(inputs: dict, *, mtp: bool) -> int:
    layers = inputs["full_attention_layers"] + int(mtp)
    return layers * 2 * inputs["kv_heads"] * inputs["head_dim"] * 2


def linear_state_bytes(inputs: dict) -> int:
    recurrent = inputs["linear_value_heads"] * inputs["linear_key_dim"] * inputs["linear_value_dim"] * 4
    convolution = inputs["linear_conv_width"] * (inputs["linear_conv_kernel"] - 1) * 2
    return inputs["linear_attention_layers"] * (recurrent + convolution)


def model_state_bytes(parameters: int, *, tp: int, dp_cp: int) -> float:
    """BF16 params, FP32 accumulated grads, distributed FP32 Adam: (6 + 12/d) bytes/param."""
    if min(parameters, tp, dp_cp) <= 0:
        raise ValueError("Parameter and parallel sizes must be positive")
    return parameters * (6 + 12 / dp_cp) / tp


def memory_estimates(config: dict, inputs: dict) -> dict:
    args = config["miles_args"]
    mtp = args["enable_mtp_training"]
    parameters = inputs["text_parameters"] + (inputs["mtp_parameters"] if mtp else 0)
    sequence = args["rollout_max_context_len"]
    full_checkpoint = sum(inputs[key] for key in ("text_parameters", "mtp_parameters", "vision_parameters")) * 2
    kv_sequence = kv_bytes_per_token(inputs, mtp=mtp) * sequence
    state = linear_state_bytes(inputs)
    engine_gpus = args["rollout_num_gpus_per_engine"]
    budget = engine_gpus * inputs["gpu_memory_bytes"] * args["sglang_mem_fraction_static"]
    requests = args["sglang_max_running_requests"]
    engine_known = full_checkpoint + requests * (kv_sequence + state)
    engine_planned = engine_known + inputs["engine_workspace_reserve_bytes"]
    layouts = []
    for gpus, tp, cp in ((1, 1, 1), (2, 2, 1), (4, 2, 2)):
        dp = gpus // (tp * cp)
        checkpoints = (inputs["layers"] + int(mtp)) * sequence * inputs["hidden_size"] * 2 / (tp * cp)
        layouts.append(
            {
                "gpus": gpus,
                "tp": tp,
                "cp": cp,
                "dp": dp,
                "model_grad_adam_gib_per_gpu": model_state_bytes(parameters, tp=tp, dp_cp=dp * cp) / GIB,
                "saved_hidden_checkpoints_gib_per_gpu": checkpoints / GIB,
                "materialized_bf16_logits_gib_per_gpu": sequence * inputs["vocab_size"] * 2 / (tp * cp * GIB),
                "one_bf16_logits_tile_mib_per_gpu": args["log_probs_chunk_size"]
                * inputs["vocab_size"]
                * 2
                / (tp * 2**20),
            }
        )
    return {
        "train_parameters_with_selected_mtp": parameters,
        "checkpoint_bf16_weights_gib": full_checkpoint / GIB,
        "kv_bytes_per_token_without_mtp": kv_bytes_per_token(inputs, mtp=False),
        "kv_bytes_per_token_with_selected_mtp": kv_bytes_per_token(inputs, mtp=mtp),
        "kv_gib_per_full_context_trajectory": kv_sequence / GIB,
        "linear_recurrent_and_conv_mib_per_trajectory": state / 2**20,
        "engine_count": args["rollout_num_gpus"] // engine_gpus,
        "engine_gpus": engine_gpus,
        "requests_per_engine": requests,
        "engine_static_budget_gib": budget / GIB,
        "engine_known_components_gib": engine_known / GIB,
        "engine_with_explicit_workspace_reserve_gib": engine_planned / GIB,
        "engine_remaining_planning_headroom_gib": (budget - engine_planned) / GIB,
        "training_layouts": layouts,
        "limits": "Not measured peak memory. Activations inside recomputed layers, allocator/communication buffers, prefix-state checkpoints, speculative buffers and engine workspaces require profiling. The 20 GiB engine reserve is an assumption.",
    }


def estimated_training_flops(config: dict, inputs: dict, total_tokens: int) -> float:
    args = config["miles_args"]
    mtp = args["enable_mtp_training"]
    # The tied output matrix is projected a second time by the MTP loss.
    gemm_parameters = inputs["text_parameters"] + (
        inputs["mtp_parameters"] + inputs["tied_embedding_parameters"] if mtp else 0
    )
    attention_layers = inputs["full_attention_layers"] + int(mtp)
    attention_forward = (
        2 * attention_layers * inputs["attention_heads"] * inputs["head_dim"] * total_tokens * (total_tokens + 1)
    )
    # Forward + backward + one full recomputation. This excludes GDN recurrence and communication.
    return 8 * gemm_parameters * total_tokens + 4 * attention_forward


def compute_scenarios(config: dict, inputs: dict) -> list[dict]:
    args = config["miles_args"]
    rows = []
    for output_tokens in (4096, 16384, 90000):
        flops_per_sample = estimated_training_flops(config, inputs, output_tokens + 2000)
        for batch in (256, args["global_batch_size"]):
            tokens = batch * output_tokens
            row = {
                "output_tokens_per_sample": output_tokens,
                "prompt_tokens_assumed": 2000,
                "global_batch": batch,
                "output_tokens_per_update": tokens,
                "training_pflop_per_sample": flops_per_sample / 1e15,
            }
            for tps in (1000, 4000, 8000):
                row[f"rollout_hours_at_aggregate_{tps}_tokens_s"] = tokens / tps / 3600
            for mfu in (0.1, 0.3):
                peak = args["actor_num_gpus_per_node"] * inputs["gpu_peak_bf16_flops"]
                row[f"train_hours_at_assumed_{int(mfu*100)}pct_peak"] = batch * flops_per_sample / (peak * mfu * 3600)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.proposed.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    inputs = model_inputs(config, base_dir=args.config.parent)
    result = {
        "memory": memory_estimates(config, inputs),
        "compute_scenarios": compute_scenarios(config, inputs),
        "compute_limitations": "Sensitivity calculations, not throughput predictions. Aggregate rollout tokens/s and training utilization are assumed; GDN recurrence, network, judge latency, speculative effects and pipeline bubbles are not modeled.",
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
