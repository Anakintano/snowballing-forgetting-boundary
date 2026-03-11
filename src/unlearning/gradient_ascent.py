"""
Gradient Ascent Unlearning for LLMs.
# Ref: Thudi et al. 2022 - Unrolling SGD

Negates the cross-entropy loss on the forget set to push the model
away from memorized knowledge. Uses LoRA for parameter-efficient unlearning.
"""

import argparse
import json
import os
import time
from datetime import datetime

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def print_vram():
    """Print current GPU status."""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        used = torch.cuda.memory_allocated() / 1e9
        print(f"VRAM total: {total:.2f} GB | used: {used:.2f} GB | free: {total - used:.2f} GB")
    else:
        print("WARNING: CUDA not available — running on CPU (will be very slow)")


def load_model_and_tokenizer(model_name):
    """Load model in 4-bit quantization with LoRA adapter.

    # HARDWARE NOTE: 4-bit NF4 quantization keeps LLaMA-3.2-1B at ~0.9GB VRAM.
    # LoRA adds negligible overhead (~0.5M trainable params).
    """
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
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # RESEARCH NOTE: LoRA on q_proj and v_proj is standard for unlearning —
    # modifying attention projections is sufficient to shift output distribution.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    return model, tokenizer


def tokenize_qa(samples, tokenizer, max_length=256):
    """Tokenize question-answer pairs into causal LM format."""
    texts = [
        f"Question: {q}\nAnswer: {a}"
        for q, a in zip(samples["question"], samples["answer"])
    ]
    return tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )


def compute_perplexity(model, dataloader, device):
    """Compute perplexity on a dataset.

    # RESEARCH NOTE: Perplexity on the forget set is our primary metric.
    # Higher perplexity after unlearning = model has forgotten those examples.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # Count non-masked tokens
            n_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def train_ga(model, forget_loader, optimizer, device, epochs, grad_accum_steps=4):
    """Run Gradient Ascent unlearning.

    # RESEARCH NOTE: GA negates the loss — instead of minimizing CE(model, forget_data),
    # we MAXIMIZE it. This pushes the model's distribution away from the forget set.
    # Simple but effective baseline. Known weakness: can degrade retain set performance.
    """
    model.train()
    global_step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(forget_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # RESEARCH NOTE: Negate the loss — this is the core of Gradient Ascent.
            # Normal training minimizes loss; we maximize it to "unlearn".
            loss = -outputs.loss / grad_accum_steps
            loss.backward()

            epoch_loss += outputs.loss.item()

            if (step + 1) % grad_accum_steps == 0:
                # HARDWARE NOTE: Gradient clipping prevents instability from negated gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}", "step": global_step})

        # Handle remaining gradients
        if (step + 1) % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(len(forget_loader), 1)
        print(f"Epoch {epoch+1}/{epochs} — avg loss (before negation): {avg_loss:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Gradient Ascent Unlearning")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="models/checkpoints/ga")
    parser.add_argument("--dry_run", action="store_true", help="Run on 10 samples only")
    parser.add_argument("--max_samples", type=int, default=-1, help="-1 for all samples")
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    print("=" * 60)
    print("  Gradient Ascent Unlearning")
    print("  Ref: Thudi et al. 2022 - Unrolling SGD")
    print("=" * 60)
    print_vram()

    # --- Load dataset ---
    # Ref: Maini et al. 2024 - TOFU Benchmark
    print("\nLoading TOFU dataset...")
    forget_ds = load_dataset("locuslab/TOFU", "forget01", split="train")
    retain_ds = load_dataset("locuslab/TOFU", "retain99", split="train")

    if args.dry_run:
        forget_ds = forget_ds.select(range(min(10, len(forget_ds))))
        retain_ds = retain_ds.select(range(min(10, len(retain_ds))))
        print(f"DRY RUN: using {len(forget_ds)} forget, {len(retain_ds)} retain samples")
    elif args.max_samples > 0:
        forget_ds = forget_ds.select(range(min(args.max_samples, len(forget_ds))))
        print(f"Using {len(forget_ds)} forget samples")

    print(f"Forget set: {len(forget_ds)} samples | Retain set: {len(retain_ds)} samples")

    # --- Load model ---
    print(f"\nLoading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    device = next(model.parameters()).device
    print_vram()

    # --- Tokenize ---
    forget_tokenized = forget_ds.map(
        lambda x: tokenize_qa(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=forget_ds.column_names,
    )
    forget_tokenized.set_format("torch")
    # HARDWARE NOTE: batch_size=1 is mandatory for 4GB VRAM
    forget_loader = DataLoader(forget_tokenized, batch_size=1, shuffle=True)

    # Also prepare forget set loader for perplexity eval (no shuffle)
    forget_eval_loader = DataLoader(forget_tokenized, batch_size=1, shuffle=False)

    # --- Baseline perplexity ---
    print("\nComputing BASELINE forget-set perplexity...")
    ppl_before = compute_perplexity(model, forget_eval_loader, device)
    print(f"Forget perplexity BEFORE unlearning: {ppl_before:.2f}")

    # --- Train (Gradient Ascent) ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    print(f"\nStarting Gradient Ascent unlearning ({args.epochs} epochs)...")
    print_vram()
    model = train_ga(model, forget_loader, optimizer, device, args.epochs, args.grad_accum_steps)

    # --- Post-unlearning perplexity ---
    print("\nComputing POST-UNLEARNING forget-set perplexity...")
    ppl_after = compute_perplexity(model, forget_eval_loader, device)
    print(f"Forget perplexity AFTER unlearning: {ppl_after:.2f}")
    print(f"Perplexity ratio: {ppl_after / max(ppl_before, 1e-8):.2f}x")

    # --- Save checkpoint ---
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nCheckpoint saved to {args.output_dir}")

    # --- Save results ---
    results_dir = "results/phase1"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "method": "gradient_ascent",
        "model": args.model_name,
        "epochs": args.epochs,
        "lr": args.lr,
        "forget_samples": len(forget_ds),
        "ppl_before": ppl_before,
        "ppl_after": ppl_after,
        "ppl_ratio": ppl_after / max(ppl_before, 1e-8),
        "timestamp": timestamp,
        "dry_run": args.dry_run,
    }
    results_path = os.path.join(results_dir, f"ga_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    print("\n" + "=" * 60)
    print(f"  RESULT: Forget perplexity {ppl_before:.2f} -> {ppl_after:.2f} ({ppl_after/max(ppl_before,1e-8):.2f}x)")
    print("=" * 60)


if __name__ == "__main__":
    main()
