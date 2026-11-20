"""Compare the frozen serving policy with a standard Transformers forward pass.

This uses the approved question and an already completed thinking prefix. It is
a numerical implementation check, with no rubric calls or optimizer updates.
"""

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path

from experiments.tcs_async_rl.amd_preflight import load_configs
from experiments.tcs_async_rl.group_inference import load_input


def compare_logprobs(served, reference, *, chosen, reference_best, tolerance):
    if not served or chosen not in served:
        raise ValueError("Serving reply lacks the chosen token's log probability")
    values = [*served.values(), *reference.values()]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite reference or served log probability")
    errors = {token: served[token] - reference[token] for token in served}
    maximum = max(abs(value) for value in errors.values())
    greedy_gap = reference[reference_best] - reference[chosen]
    return {
        "maximum_top_logprob_error_nats": maximum,
        "reference_greedy_gap_nats": greedy_gap,
        "same_greedy_token": chosen == reference_best,
        "passed": maximum <= tolerance and greedy_gap <= tolerance,
        "errors_nats": errors,
    }


def summarize_logits(logits, output, tokenizer, *, tolerance):
    # Torch is deferred because comparison controls run on the CPU-only VM.
    import torch

    logprobs = torch.log_softmax(logits.float(), dim=-1)
    best = int(logprobs.argmax())
    reference = {token: float(logprobs[token]) for token in {*output["logprobs"], best}}
    return {
        "hf_greedy_token_id": best,
        "hf_greedy_text": tokenizer.decode([best], skip_special_tokens=False),
        **compare_logprobs(
            output["logprobs"], reference, chosen=output["chosen"], reference_best=best, tolerance=tolerance
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    cli = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Reference forward passes require the approved GPU allocation")
    os.environ["VLLM_NO_USAGE_STATS"] = "1"
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    from vllm import LLM, SamplingParams

    config, training = load_configs(cli.config)
    group, _ = load_configs(cli.config.parent / config["group_config"])
    row = load_input(group, training)
    fixture_bytes = Path(config["fixture_file"]).read_bytes()
    if hashlib.sha256(fixture_bytes).hexdigest() != config["fixture_sha256"]:
        raise ValueError("Completed-response fixture changed")
    fixture = json.loads(fixture_bytes)
    controls = config["technical_controls"]
    tokenizer = AutoTokenizer.from_pretrained(cli.model_dir, local_files_only=True)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": row["problem"]}],
        tokenize=False,
        add_generation_prompt=True,
        **training["miles_args"]["apply_chat_template_kwargs"],
    )
    if rendered != row["formatted_prompt"]:
        raise ValueError("Reference prompt differs from the approved prompt")
    prompt = tokenizer.encode(rendered, add_special_tokens=False)
    continuation = tokenizer.encode(fixture["response_text"], add_special_tokens=False)
    if len(continuation) < max(controls["prefix_lengths"]):
        raise ValueError("Completed-response fixture is shorter than the declared prefixes")
    cases = [prompt + continuation[:length] for length in controls["prefix_lengths"]]
    (cli.run_dir / "reference-inputs.json").write_text(json.dumps(cases) + "\n")
    (cli.run_dir / "formatted-prompt.txt").write_text(rendered)
    (cli.run_dir / "reference-prefix-texts.json").write_text(
        json.dumps([tokenizer.decode(tokens, skip_special_tokens=False) for tokens in cases], ensure_ascii=False)
        + "\n"
    )
    engine = LLM(
        model=str(cli.model_dir),
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=training["miles_args"]["rollout_max_context_len"],
        language_model_only=True,
        enforce_eager=True,
        gpu_memory_utilization=controls["gpu_memory_utilization"],
        max_num_seqs=1,
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        generation_config="vllm",
        logprobs_mode="raw_logprobs",
        seed=group["inference"]["seed"],
    )
    try:
        outputs = engine.generate(
            [{"prompt_token_ids": tokens} for tokens in cases],
            SamplingParams(temperature=0, max_tokens=1, logprobs=20, detokenize=False),
            use_tqdm=False,
        )
        served = []
        for tokens, output in zip(cases, outputs, strict=True):
            if output.prompt_token_ids != tokens or len(output.outputs[0].token_ids) != 1:
                raise ValueError("Reference request changed token IDs or decode length")
            completion = output.outputs[0]
            served.append(
                {
                    "chosen": completion.token_ids[0],
                    "logprobs": {token: value.logprob for token, value in completion.logprobs[0].items()},
                }
            )
        (cli.run_dir / "vllm-reference-logprobs.json").write_text(json.dumps(served, indent=2) + "\n")
    finally:
        engine.llm_engine.engine_core.shutdown(timeout=20)
    del engine
    gc.collect()
    model, loading = AutoModelForImageTextToText.from_pretrained(
        cli.model_dir,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="eager",
        output_loading_info=True,
    )
    (cli.run_dir / "hf-loading.json").write_text(
        json.dumps(loading, indent=2, default=lambda value: sorted(value) if isinstance(value, set) else str(value))
        + "\n"
    )
    if loading.get("missing_keys"):
        raise ValueError("Transformers initialized missing checkpoint parameters")
    model.eval().to("cuda")
    results = []
    precision_results = []
    captured = []
    precision_diagnostic = controls.get("diagnose_precision", False)
    head = model.get_output_embeddings()
    hook = (
        head.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
        if precision_diagnostic
        else None
    )
    with torch.inference_mode():
        for tokens, output in zip(cases, served, strict=True):
            captured.clear()
            logits = model(input_ids=torch.tensor([tokens], device="cuda"), use_cache=False, logits_to_keep=1).logits[
                0, -1
            ]
            results.append(
                {
                    "prompt_tokens": len(tokens),
                    "served_greedy_token_id": output["chosen"],
                    "served_greedy_text": tokenizer.decode([output["chosen"]], skip_special_tokens=False),
                    **summarize_logits(logits, output, tokenizer, tolerance=controls["tolerance_nats"]),
                }
            )
            print(json.dumps({key: value for key, value in results[-1].items() if key != "errors_nats"}), flush=True)
            if precision_diagnostic:
                if len(captured) != 1 or captured[0].shape[-2] != 1:
                    raise ValueError("Precision diagnostic requires one captured final-token head input")
                # Reuse identical BF16 hidden states. Only this diagnostic head calculation changes precision.
                precise_logits = torch.nn.functional.linear(
                    captured[0].float(), head.weight.float(), None if head.bias is None else head.bias.float()
                )[0, -1]
                precision_results.append(
                    {
                        "prompt_tokens": len(tokens),
                        "hf_bf16_hidden_fp32_head": summarize_logits(
                            precise_logits, output, tokenizer, tolerance=controls["tolerance_nats"]
                        ),
                    }
                )
    if hook is not None:
        hook.remove()
        captured.clear()
        # Widen the same loaded checkpoint values. This does not restore precision absent in the checkpoint.
        model.float()
        with torch.inference_mode():
            for tokens, output, result in zip(cases, served, precision_results, strict=True):
                logits = model(
                    input_ids=torch.tensor([tokens], device="cuda"), use_cache=False, logits_to_keep=1
                ).logits[0, -1]
                result["hf_fp32_model"] = summarize_logits(
                    logits, output, tokenizer, tolerance=controls["tolerance_nats"]
                )
        (cli.run_dir / "precision-diagnostic.json").write_text(json.dumps(precision_results, indent=2) + "\n")
    summary = {
        "passed": all(case["passed"] for case in results),
        "cases": results,
        "controls": controls,
        "input_sha256": group["input_sha256"],
        "fixture_sha256": config["fixture_sha256"],
        "scope": "Three short teacher-forced prefixes, BF16 weights and eager HF attention. Not 90k decode parity or Megatron parity.",
    }
    (cli.run_dir / "policy-reference-results.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["passed"]:
        raise ValueError("Serving/reference numerical comparison failed; inspect saved metrics")


if __name__ == "__main__":
    main()
