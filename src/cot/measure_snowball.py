"""
Measure per-step CoT error rates and snowball effect.

Compares base model vs unlearned model CoT chains:
- Per-step accuracy (does the step contain correct information?)
- Cumulative error probability as a function of chain depth
- Boundary vs control chain comparison

Uses the generated chains from generate_chains.py as input.
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def print_vram():
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        used = torch.cuda.memory_allocated() / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {used:.2f}/{total:.2f} GB")


def load_model(model_name, adapter_path=None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map={"": 0},
    )
    if adapter_path and os.path.exists(adapter_path):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def compute_step_perplexity(model, tokenizer, prompt_text, step_text):
    """Compute perplexity of a specific CoT step given its prompt context.

    RESEARCH NOTE: Per-step perplexity measures how "surprised" the model is
    at each reasoning step. Post-unlearning, we expect higher perplexity on
    boundary-adjacent steps — the model struggles to generate coherent reasoning
    through partially-erased knowledge.
    """
    # Encode prompt + step together
    full_text = prompt_text + "\n" + step_text
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=768)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Get prompt length to only compute loss on the step part
    prompt_tokens = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512)
    prompt_len = prompt_tokens["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Compute loss only on step tokens (after prompt)
    shift_logits = logits[:, prompt_len:-1, :].contiguous()
    shift_labels = inputs["input_ids"][:, prompt_len+1:].contiguous()

    if shift_logits.shape[1] == 0 or shift_labels.shape[1] == 0:
        return float("nan")

    # Align lengths
    min_len = min(shift_logits.shape[1], shift_labels.shape[1])
    shift_logits = shift_logits[:, :min_len, :]
    shift_labels = shift_labels[:, :min_len]

    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return torch.exp(loss).item()


def measure_chain_perplexity(model, tokenizer, chain_data):
    """Measure per-step perplexity for a CoT chain.

    Returns list of per-step perplexity values.
    """
    prompt = chain_data["prompt"]
    steps = chain_data.get("parsed_steps", [])
    generated = chain_data.get("generated_text", "")

    if not steps:
        # Fall back: split generated text into rough steps
        steps = [s.strip() for s in generated.split("\n") if s.strip() and len(s.strip()) > 10]

    step_ppls = []
    context = prompt
    for step in steps:
        ppl = compute_step_perplexity(model, tokenizer, context, step)
        step_ppls.append(ppl)
        context = context + "\n" + step  # accumulate context

    return step_ppls


def compute_snowball_metrics(boundary_ppls, control_ppls):
    """Compute snowball effect metrics.

    RESEARCH NOTE: The snowball effect is measured as the ratio of
    per-step perplexity increase at deeper chain positions for boundary
    chains vs control chains. If partial unlearning causes snowballing,
    boundary chains should show steeper perplexity growth with depth.
    """
    metrics = {}

    # Flatten to per-depth stats
    max_depth = max(
        max((len(p) for p in boundary_ppls), default=0),
        max((len(p) for p in control_ppls), default=0),
    )

    boundary_by_depth = {d: [] for d in range(max_depth)}
    control_by_depth = {d: [] for d in range(max_depth)}

    for ppls in boundary_ppls:
        for d, p in enumerate(ppls):
            if not np.isnan(p) and p < 1e6:  # filter outliers
                boundary_by_depth[d].append(p)

    for ppls in control_ppls:
        for d, p in enumerate(ppls):
            if not np.isnan(p) and p < 1e6:
                control_by_depth[d].append(p)

    metrics["per_depth"] = {}
    for d in range(max_depth):
        b_vals = boundary_by_depth[d]
        c_vals = control_by_depth[d]
        metrics["per_depth"][d] = {
            "boundary_mean_ppl": float(np.mean(b_vals)) if b_vals else None,
            "boundary_std_ppl": float(np.std(b_vals)) if b_vals else None,
            "control_mean_ppl": float(np.mean(c_vals)) if c_vals else None,
            "control_std_ppl": float(np.std(c_vals)) if c_vals else None,
            "boundary_n": len(b_vals),
            "control_n": len(c_vals),
        }

        if b_vals and c_vals:
            ratio = np.mean(b_vals) / max(np.mean(c_vals), 1e-8)
            metrics["per_depth"][d]["ppl_ratio"] = float(ratio)

    # Overall snowball score: slope of boundary ppl vs depth / slope of control ppl vs depth
    b_means = [np.mean(boundary_by_depth[d]) for d in range(max_depth) if boundary_by_depth[d]]
    c_means = [np.mean(control_by_depth[d]) for d in range(max_depth) if control_by_depth[d]]

    if len(b_means) >= 2:
        b_slope = (b_means[-1] - b_means[0]) / max(len(b_means) - 1, 1)
        metrics["boundary_ppl_slope"] = float(b_slope)
    if len(c_means) >= 2:
        c_slope = (c_means[-1] - c_means[0]) / max(len(c_means) - 1, 1)
        metrics["control_ppl_slope"] = float(c_slope)
    if len(b_means) >= 2 and len(c_means) >= 2:
        metrics["snowball_ratio"] = float(b_slope / max(abs(c_slope), 1e-8))

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Measure CoT snowball effect")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--chains_file", type=str, default=None,
                        help="Path to generated chains JSON (from generate_chains.py)")
    parser.add_argument("--adapter_path", type=str, default="models/checkpoints/ga")
    parser.add_argument("--output_dir", type=str, default="results/phase2")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  CoT Snowball Effect Measurement")
    print("=" * 60)
    print_vram()

    # Load chains
    if args.chains_file and os.path.exists(args.chains_file):
        print(f"\nLoading chains from {args.chains_file}")
        with open(args.chains_file, "r", encoding="utf-8") as f:
            chains = json.load(f)
    else:
        # Generate chains on the fly
        print("\nNo chains file provided — generating chains first...")
        from generate_chains import build_cot_prompts, run_generation, load_model as load_gen_model

        forget_ds = load_dataset("locuslab/TOFU", "forget01", split="train")
        retain_ds = load_dataset("locuslab/TOFU", "retain99", split="train")

        n = 3 if args.dry_run else 20
        boundary_prompts, control_prompts = build_cot_prompts(forget_ds, retain_ds, n, n)

        # Generate with base model
        model, tokenizer = load_gen_model(args.model_name)
        base_boundary = run_generation(model, tokenizer, boundary_prompts, "Base/Boundary")
        base_control = run_generation(model, tokenizer, control_prompts, "Base/Control")
        del model; torch.cuda.empty_cache()

        # Generate with unlearned model
        model, tokenizer = load_gen_model(args.model_name, args.adapter_path)
        bp2, cp2 = build_cot_prompts(forget_ds, retain_ds, n, n)
        ul_boundary = run_generation(model, tokenizer, bp2, "Unlearned/Boundary")
        ul_control = run_generation(model, tokenizer, cp2, "Unlearned/Control")
        del model; torch.cuda.empty_cache()

        chains = {
            "base_boundary": base_boundary, "base_control": base_control,
            "unlearned_boundary": ul_boundary, "unlearned_control": ul_control,
        }

    # --- Measure per-step perplexity with UNLEARNED model ---
    print(f"\nLoading unlearned model for perplexity measurement...")
    model, tokenizer = load_model(args.model_name, args.adapter_path)
    print_vram()

    print("\nMeasuring per-step perplexity on UNLEARNED model...")

    # Measure boundary chains
    ul_boundary_ppls = []
    for chain in tqdm(chains.get("unlearned_boundary", []), desc="Boundary PPL"):
        ppls = measure_chain_perplexity(model, tokenizer, chain)
        ul_boundary_ppls.append(ppls)

    # Measure control chains
    ul_control_ppls = []
    for chain in tqdm(chains.get("unlearned_control", []), desc="Control PPL"):
        ppls = measure_chain_perplexity(model, tokenizer, chain)
        ul_control_ppls.append(ppls)

    del model; torch.cuda.empty_cache()

    # --- Also measure with BASE model for comparison ---
    print(f"\nLoading base model for perplexity measurement...")
    model, tokenizer = load_model(args.model_name)
    print_vram()

    base_boundary_ppls = []
    for chain in tqdm(chains.get("base_boundary", []), desc="Base Boundary PPL"):
        ppls = measure_chain_perplexity(model, tokenizer, chain)
        base_boundary_ppls.append(ppls)

    base_control_ppls = []
    for chain in tqdm(chains.get("base_control", []), desc="Base Control PPL"):
        ppls = measure_chain_perplexity(model, tokenizer, chain)
        base_control_ppls.append(ppls)

    del model; torch.cuda.empty_cache()

    # --- Compute snowball metrics ---
    print("\n" + "=" * 60)
    print("  SNOWBALL METRICS")
    print("=" * 60)

    ul_metrics = compute_snowball_metrics(ul_boundary_ppls, ul_control_ppls)
    base_metrics = compute_snowball_metrics(base_boundary_ppls, base_control_ppls)

    print("\nUNLEARNED model — per-depth perplexity:")
    for d, vals in ul_metrics.get("per_depth", {}).items():
        b = vals.get("boundary_mean_ppl")
        c = vals.get("control_mean_ppl")
        r = vals.get("ppl_ratio")
        print(f"  Depth {d}: boundary={b:.1f}, control={c:.1f}, ratio={r:.2f}" if b and c and r
              else f"  Depth {d}: insufficient data")

    if "snowball_ratio" in ul_metrics:
        print(f"\nSnowball ratio (unlearned): {ul_metrics['snowball_ratio']:.2f}")
        print(f"  (boundary ppl slope / control ppl slope)")
        print(f"  >1.0 means boundary chains degrade faster = snowball effect")

    if "snowball_ratio" in base_metrics:
        print(f"Snowball ratio (base):      {base_metrics['snowball_ratio']:.2f}")

    # --- Save ---
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "timestamp": timestamp,
        "model": args.model_name,
        "dry_run": args.dry_run,
        "unlearned_metrics": ul_metrics,
        "base_metrics": base_metrics,
        "unlearned_boundary_ppls": ul_boundary_ppls,
        "unlearned_control_ppls": ul_control_ppls,
        "base_boundary_ppls": base_boundary_ppls,
        "base_control_ppls": base_control_ppls,
    }
    out_path = os.path.join(args.output_dir, f"snowball_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
