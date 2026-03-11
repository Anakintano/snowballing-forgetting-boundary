"""
Bad Teacher Unlearning for LLMs.
# Ref: Chundawat et al. AAAI 2023 - Bad Teaching Unlearning

Uses an "incompetent teacher" (random LoRA adapter) to generate soft labels
on the forget set. The student is trained to match the teacher's garbage output
distribution on forget data via KL divergence, while maintaining retain performance.
"""

import argparse
import json
import os
from datetime import datetime

import torch
import torch.nn.functional as F
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
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        used = torch.cuda.memory_allocated() / 1e9
        print(f"VRAM total: {total:.2f} GB | used: {used:.2f} GB | free: {total - used:.2f} GB")
    else:
        print("WARNING: CUDA not available")


def load_base_model(model_name):
    """Load 4-bit quantized base model + tokenizer (no LoRA yet)."""
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
    return model, tokenizer


def get_lora_config():
    return LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def tokenize_qa(samples, tokenizer, max_length=256):
    texts = [
        f"Question: {q}\nAnswer: {a}"
        for q, a in zip(samples["question"], samples["answer"])
    ]
    return tokenizer(texts, max_length=max_length, truncation=True,
                     padding="max_length", return_tensors="pt")


def compute_perplexity(model, dataloader, device):
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
            n_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens
    return torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item()


def precompute_teacher_logits(model, dataloader, device):
    """Generate soft labels from incompetent (random LoRA) teacher.

    # HARDWARE NOTE: We precompute ALL teacher logits and store on CPU.
    # This avoids needing two models in VRAM simultaneously.
    # For TOFU forget01 (~40 samples), this is ~40 tensors of shape [1, seq_len, vocab].
    # We store only top-k logits to save RAM.
    """
    model.eval()
    teacher_logits = []

    print("Precomputing incompetent teacher logits (random LoRA)...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Teacher forward", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # HARDWARE NOTE: Move to CPU immediately to free VRAM
            teacher_logits.append(outputs.logits.cpu())

    return teacher_logits


def train_bad_teacher(model, forget_loader, retain_loader, teacher_logits,
                      optimizer, device, epochs, alpha, temperature, grad_accum_steps=4):
    """Bad Teacher training loop.

    # RESEARCH NOTE: The loss combines two objectives:
    # 1. KL divergence on forget set: force student to match incompetent teacher's
    #    random output distribution. This "overwrites" memorized knowledge.
    # 2. Cross-entropy on retain set: maintain utility on non-forgotten data.
    # alpha controls the balance. Higher alpha = more aggressive forgetting.
    """
    model.train()
    global_step = 0

    for epoch in range(epochs):
        epoch_forget_loss = 0.0
        epoch_retain_loss = 0.0
        optimizer.zero_grad()

        # Create retain iterator (cycle if shorter than forget)
        retain_iter = iter(retain_loader)

        pbar = tqdm(enumerate(forget_loader), total=len(forget_loader),
                    desc=f"Epoch {epoch+1}/{epochs}")

        for step, forget_batch in pbar:
            # --- Forget loss: KL divergence with teacher ---
            input_ids = forget_batch["input_ids"].to(device)
            attention_mask = forget_batch["attention_mask"].to(device)

            student_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits

            # Get precomputed teacher logits for this sample
            t_logits = teacher_logits[step % len(teacher_logits)].to(device)

            # RESEARCH NOTE: Temperature scaling softens distributions before KL.
            # Higher temperature = softer distributions = more knowledge transfer.
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            teacher_probs = F.softmax(t_logits / temperature, dim=-1)

            # KL(teacher || student) — we want student to match teacher's distribution
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
            # Scale by T^2 as per distillation convention
            kl_loss = kl_loss * (temperature ** 2)

            # --- Retain loss: standard cross-entropy ---
            try:
                retain_batch = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                retain_batch = next(retain_iter)

            retain_ids = retain_batch["input_ids"].to(device)
            retain_mask = retain_batch["attention_mask"].to(device)
            retain_labels = retain_ids.clone()
            retain_labels[retain_mask == 0] = -100

            retain_outputs = model(input_ids=retain_ids, attention_mask=retain_mask,
                                   labels=retain_labels)
            ce_loss = retain_outputs.loss

            # --- Combined loss ---
            loss = (alpha * kl_loss + (1 - alpha) * ce_loss) / grad_accum_steps
            loss.backward()

            epoch_forget_loss += kl_loss.item()
            epoch_retain_loss += ce_loss.item()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            pbar.set_postfix({
                "kl": f"{kl_loss.item():.3f}",
                "ce": f"{ce_loss.item():.3f}",
                "step": global_step,
            })

        if (step + 1) % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        n = max(len(forget_loader), 1)
        print(f"Epoch {epoch+1} — KL: {epoch_forget_loss/n:.4f}, CE: {epoch_retain_loss/n:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Bad Teacher Unlearning")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for forget KL loss")
    parser.add_argument("--temperature", type=float, default=4.0, help="KL temperature")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints/bt")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    print("=" * 60)
    print("  Bad Teacher Unlearning")
    print("  Ref: Chundawat et al. AAAI 2023")
    print("=" * 60)
    print_vram()

    # --- Load dataset ---
    print("\nLoading TOFU dataset...")
    forget_ds = load_dataset("locuslab/TOFU", "forget01", split="train")
    retain_ds = load_dataset("locuslab/TOFU", "retain99", split="train")

    if args.dry_run:
        forget_ds = forget_ds.select(range(min(10, len(forget_ds))))
        retain_ds = retain_ds.select(range(min(10, len(retain_ds))))
        print(f"DRY RUN: {len(forget_ds)} forget, {len(retain_ds)} retain")
    elif args.max_samples > 0:
        forget_ds = forget_ds.select(range(min(args.max_samples, len(forget_ds))))

    print(f"Forget: {len(forget_ds)} | Retain: {len(retain_ds)}")

    # --- Load model + random LoRA (incompetent teacher) ---
    # HARDWARE NOTE: We load ONE model and swap LoRA adapters to stay under 4GB.
    # Step 1: random LoRA = incompetent teacher → precompute logits
    # Step 2: fresh LoRA = student → train
    print(f"\nLoading model: {args.model_name}")
    model, tokenizer = load_base_model(args.model_name)
    device = next(model.parameters()).device

    # Tokenize
    forget_tok = forget_ds.map(lambda x: tokenize_qa(x, tokenizer, args.max_length),
                               batched=True, remove_columns=forget_ds.column_names)
    forget_tok.set_format("torch")
    retain_tok = retain_ds.map(lambda x: tokenize_qa(x, tokenizer, args.max_length),
                               batched=True, remove_columns=retain_ds.column_names)
    retain_tok.set_format("torch")

    forget_loader = DataLoader(forget_tok, batch_size=1, shuffle=False)
    retain_loader = DataLoader(retain_tok, batch_size=1, shuffle=True)
    forget_eval_loader = DataLoader(forget_tok, batch_size=1, shuffle=False)

    # --- Step 1: Incompetent teacher (random LoRA) ---
    # RESEARCH NOTE: Instead of a separate randomly-initialized model,
    # we use a random (untrained) LoRA adapter. This produces near-random
    # output distributions — acting as the "incompetent teacher".
    print("\nAttaching random LoRA adapter (incompetent teacher)...")
    model = get_peft_model(model, get_lora_config())
    model.gradient_checkpointing_enable()
    print_vram()

    # Precompute teacher logits with random LoRA
    teacher_logits = precompute_teacher_logits(model, forget_loader, device)
    print(f"Cached {len(teacher_logits)} teacher logit tensors on CPU")

    # --- Baseline perplexity (with random LoRA — should be high) ---
    # We need perplexity of the BASE model, so disable adapter
    model.disable_adapter_layers()
    print("\nComputing BASELINE forget-set perplexity (base model)...")
    ppl_before = compute_perplexity(model, forget_eval_loader, device)
    print(f"Forget perplexity BEFORE unlearning: {ppl_before:.2f}")

    # --- Step 2: Re-enable adapter with fresh random init for student ---
    # RESEARCH NOTE: We reinitialize the LoRA weights for the student.
    # The student starts from the base model's behavior (LoRA=0 would be identity,
    # but we use small random init which is near-identity).
    model.enable_adapter_layers()
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            torch.nn.init.normal_(param, mean=0.0, std=0.01)

    model.print_trainable_parameters()

    # --- Train ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    forget_train_loader = DataLoader(forget_tok, batch_size=1, shuffle=True)

    print(f"\nStarting Bad Teacher training ({args.epochs} epochs, alpha={args.alpha})...")
    model = train_bad_teacher(
        model, forget_train_loader, retain_loader, teacher_logits,
        optimizer, device, args.epochs, args.alpha, args.temperature, args.grad_accum_steps
    )

    # --- Post-unlearning perplexity ---
    print("\nComputing POST-UNLEARNING forget-set perplexity...")
    ppl_after = compute_perplexity(model, forget_eval_loader, device)
    print(f"Forget perplexity AFTER unlearning: {ppl_after:.2f}")
    print(f"Perplexity ratio: {ppl_after / max(ppl_before, 1e-8):.2f}x")

    # --- Save ---
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Checkpoint saved to {args.output_dir}")

    results_dir = "results/phase1"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        "method": "bad_teacher",
        "model": args.model_name,
        "epochs": args.epochs,
        "lr": args.lr,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "forget_samples": len(forget_ds),
        "ppl_before": ppl_before,
        "ppl_after": ppl_after,
        "ppl_ratio": ppl_after / max(ppl_before, 1e-8),
        "timestamp": timestamp,
        "dry_run": args.dry_run,
    }
    results_path = os.path.join(results_dir, f"bt_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    print("\n" + "=" * 60)
    print(f"  RESULT: Forget perplexity {ppl_before:.2f} -> {ppl_after:.2f} ({ppl_after/max(ppl_before,1e-8):.2f}x)")
    print("=" * 60)


if __name__ == "__main__":
    main()
