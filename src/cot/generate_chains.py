"""
Generate CoT reasoning chains that pass through forgotten vs control concepts.
# Ref: Maini et al. 2024 - TOFU Benchmark

Creates two sets of multi-hop QA chains:
1. Boundary-adjacent: chains whose intermediate steps touch forgotten concepts
2. Control: chains that stay entirely in retain-set territory

Each chain step is labeled: clean / boundary-adjacent / boundary-crossing
"""

import argparse
import json
import os
from datetime import datetime

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
    """Load base or unlearned model in 4-bit."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},  # HARDWARE NOTE: device_map="auto" segfaults on Windows
    )

    if adapter_path and os.path.exists(adapter_path):
        from peft import PeftModel
        print(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def build_cot_prompts(forget_ds, retain_ds, n_boundary=20, n_control=20):
    """Build CoT prompts that either cross or avoid the forget boundary.

    RESEARCH NOTE: We construct multi-hop chains by chaining TOFU QA pairs.
    - Boundary chains: start with retain question, then ask about a forgotten entity
      in the reasoning steps, forcing the model to reason through forgotten knowledge.
    - Control chains: use only retain-set entities throughout.

    Each prompt asks the model to reason step-by-step, creating a CoT trace
    we can analyze for error propagation.
    """
    boundary_prompts = []
    control_prompts = []

    # Extract forget-set entities (author names from TOFU)
    forget_questions = [s["question"] for s in forget_ds]
    forget_answers = [s["answer"] for s in forget_ds]
    retain_questions = [s["question"] for s in retain_ds]
    retain_answers = [s["answer"] for s in retain_ds]

    # --- Boundary-adjacent chains ---
    # RESEARCH NOTE: These chains START with a retain-set question but then
    # require reasoning about a forget-set entity. This tests whether the model
    # can maintain coherent CoT when passing through partially-erased knowledge.
    for i in range(min(n_boundary, len(forget_questions))):
        # Pick a retain question as context, then ask about forgotten entity
        retain_idx = i % len(retain_questions)
        prompt = (
            f"Answer the following questions step by step. Show your reasoning for each step.\n\n"
            f"Step 1: {retain_questions[retain_idx]}\n"
            f"Step 2: Based on your answer above, now answer: {forget_questions[i]}\n"
            f"Step 3: Combine the information from Steps 1 and 2 to explain "
            f"how these facts relate to each other.\n\n"
            f"Think step by step:"
        )
        boundary_prompts.append({
            "prompt": prompt,
            "type": "boundary",
            "forget_question": forget_questions[i],
            "forget_answer": forget_answers[i],
            "retain_question": retain_questions[retain_idx],
            "retain_answer": retain_answers[retain_idx],
            "chain_depth": 3,
        })

    # --- Control chains (retain-only) ---
    for i in range(min(n_control, len(retain_questions) - 1)):
        j = (i + 1) % len(retain_questions)
        prompt = (
            f"Answer the following questions step by step. Show your reasoning for each step.\n\n"
            f"Step 1: {retain_questions[i]}\n"
            f"Step 2: Based on your answer above, now answer: {retain_questions[j]}\n"
            f"Step 3: Combine the information from Steps 1 and 2 to explain "
            f"how these facts relate to each other.\n\n"
            f"Think step by step:"
        )
        control_prompts.append({
            "prompt": prompt,
            "type": "control",
            "retain_question_1": retain_questions[i],
            "retain_answer_1": retain_answers[i],
            "retain_question_2": retain_questions[j],
            "retain_answer_2": retain_answers[j],
            "chain_depth": 3,
        })

    return boundary_prompts, control_prompts


def generate_chain(model, tokenizer, prompt, max_new_tokens=300):
    """Generate a CoT chain from a prompt. Returns generated text."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # RESEARCH NOTE: greedy decoding for reproducibility
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated part
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text


def run_generation(model, tokenizer, prompts, desc="Generating"):
    """Generate CoT chains for all prompts."""
    results = []
    for item in tqdm(prompts, desc=desc):
        text = generate_chain(model, tokenizer, item["prompt"])
        item["generated_text"] = text
        # Split into steps for per-step analysis
        steps = []
        for line in text.split("\n"):
            line = line.strip()
            if line and (line.lower().startswith("step") or line.startswith("1") or
                        line.startswith("2") or line.startswith("3") or len(line) > 20):
                steps.append(line)
        item["parsed_steps"] = steps
        item["n_steps_generated"] = len(steps)
        results.append(item)
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate CoT chains for snowball analysis")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to unlearned LoRA adapter (if None, uses base model)")
    parser.add_argument("--n_boundary", type=int, default=20)
    parser.add_argument("--n_control", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="results/phase2")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--mode", type=str, default="base", choices=["base", "unlearned"],
                        help="Run base or unlearned model (separate invocations to avoid Windows segfault)")
    args = parser.parse_args()

    print("=" * 60)
    print("  CoT Chain Generation for Snowball Analysis")
    print("=" * 60)
    print_vram()

    # Load dataset
    print("\nLoading TOFU dataset...")
    forget_ds = load_dataset("locuslab/TOFU", "forget01", split="train")
    retain_ds = load_dataset("locuslab/TOFU", "retain99", split="train")

    if args.dry_run:
        args.n_boundary = 3
        args.n_control = 3
        print(f"DRY RUN: {args.n_boundary} boundary + {args.n_control} control chains")

    # Build prompts
    boundary_prompts, control_prompts = build_cot_prompts(
        forget_ds, retain_ds, args.n_boundary, args.n_control
    )
    print(f"Built {len(boundary_prompts)} boundary + {len(control_prompts)} control prompts")

    # --- Load model based on mode ---
    adapter = args.adapter_path or "models/checkpoints/ga"
    if args.mode == "unlearned":
        if not os.path.exists(adapter):
            print(f"ERROR: No adapter at {adapter}. Run gradient_ascent.py first.")
            return
        print(f"\nLoading UNLEARNED model with adapter: {adapter}")
        model, tokenizer = load_model(args.model_name, adapter)
    else:
        print(f"\nLoading BASE model: {args.model_name}")
        model, tokenizer = load_model(args.model_name)
    print_vram()

    print(f"\n--- {args.mode.upper()} MODEL: Boundary chains ---")
    boundary_results = run_generation(model, tokenizer, boundary_prompts, f"{args.mode}/Boundary")
    print(f"--- {args.mode.upper()} MODEL: Control chains ---")
    control_results = run_generation(model, tokenizer, control_prompts, f"{args.mode}/Control")

    # --- Save results ---
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "model": args.model_name,
        "mode": args.mode,
        "adapter": adapter if args.mode == "unlearned" else None,
        "n_boundary": len(boundary_results),
        "n_control": len(control_results),
        "timestamp": timestamp,
        "dry_run": args.dry_run,
        f"{args.mode}_boundary": boundary_results,
        f"{args.mode}_control": control_results,
    }
    out_path = os.path.join(args.output_dir, f"cot_chains_{args.mode}_{timestamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    # Quick summary
    avg_b_steps = sum(r["n_steps_generated"] for r in boundary_results) / max(len(boundary_results), 1)
    avg_c_steps = sum(r["n_steps_generated"] for r in control_results) / max(len(control_results), 1)
    print(f"\n{args.mode} model avg steps — boundary: {avg_b_steps:.1f}, control: {avg_c_steps:.1f}")
    print(f"Run with --mode {'unlearned' if args.mode == 'base' else 'base'} to get comparison.")


if __name__ == "__main__":
    main()
