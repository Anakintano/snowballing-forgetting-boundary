"""
Negative Preference Optimization (NPO) for LLM Unlearning.
# Ref: Zhang et al. 2024 - Negative Preference Optimization

Adapts DPO for unlearning: treats forget-set answers as dispreferred responses
and pushes model probability below the reference model's probability.
Reference model = base model (LoRA disabled), policy = base + LoRA.
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


def load_model_and_tokenizer(model_name):
    """Load 4-bit model with LoRA."""
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

    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    return model, tokenizer


def tokenize_qa(samples, tokenizer, max_length=256):
    texts = [f"Question: {q}\nAnswer: {a}"
             for q, a in zip(samples["question"], samples["answer"])]
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


def compute_sequence_log_probs(model, input_ids, attention_mask, labels):
    """Compute per-sequence average log probability.

    Returns a scalar: mean log P(token | context) over non-masked tokens.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Shift for causal LM: predict next token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    # Gather log probs for actual tokens
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(2)).squeeze(2)

    # Mask padding
    mask = (shift_labels != -100).float()
    seq_log_prob = (token_log_probs * mask).sum() / mask.sum().clamp(min=1)
    return seq_log_prob


def precompute_reference_log_probs(model, dataloader, device):
    """Precompute reference model log-probs on forget set.

    # HARDWARE NOTE: We disable LoRA adapters to get base model behavior,
    # compute all log-probs, store on CPU. This avoids needing two models in VRAM.
    # For TOFU forget01 (~40 samples), CPU storage is negligible.
    """
    model.eval()
    model.disable_adapter_layers()

    ref_log_probs = []
    print("Precomputing reference (base model) log-probs on forget set...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Ref log-probs", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            log_prob = compute_sequence_log_probs(model, input_ids, attention_mask, labels)
            ref_log_probs.append(log_prob.cpu())

    model.enable_adapter_layers()
    return ref_log_probs


def train_npo(model, forget_loader, retain_loader, ref_log_probs,
              optimizer, device, epochs, beta, gamma, grad_accum_steps=4):
    """NPO training loop.

    # RESEARCH NOTE: NPO loss pushes model's log-prob on forget data BELOW the
    # reference model's log-prob. This is more controlled than raw Gradient Ascent
    # because the reference anchors the optimization — preventing catastrophic drift.
    #
    # NPO_loss = -log(sigmoid(-beta * (log_pi - log_ref)))
    # where log_pi = policy (current model), log_ref = reference (precomputed base)
    #
    # Intuitively: we want log_pi << log_ref on forget data, i.e., the model
    # assigns LOWER probability than the original model to forgotten content.
    """
    model.train()
    global_step = 0

    for epoch in range(epochs):
        epoch_npo_loss = 0.0
        epoch_retain_loss = 0.0
        optimizer.zero_grad()
        retain_iter = iter(retain_loader)

        pbar = tqdm(enumerate(forget_loader), total=len(forget_loader),
                    desc=f"Epoch {epoch+1}/{epochs}")

        for step, forget_batch in pbar:
            # --- NPO loss on forget set ---
            input_ids = forget_batch["input_ids"].to(device)
            attention_mask = forget_batch["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            policy_log_prob = compute_sequence_log_probs(
                model, input_ids, attention_mask, labels
            )
            ref_log_prob = ref_log_probs[step % len(ref_log_probs)].to(device)

            # NPO loss: -log(sigmoid(-beta * (log_pi - log_ref)))
            log_ratio = policy_log_prob - ref_log_prob
            npo_loss = -F.logsigmoid(-beta * log_ratio)

            # --- Retain loss: standard CE ---
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
            total_loss = (npo_loss + gamma * ce_loss) / grad_accum_steps
            total_loss.backward()

            epoch_npo_loss += npo_loss.item()
            epoch_retain_loss += ce_loss.item()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            pbar.set_postfix({
                "npo": f"{npo_loss.item():.3f}",
                "ce": f"{ce_loss.item():.3f}",
                "step": global_step,
            })

        if (step + 1) % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        n = max(len(forget_loader), 1)
        print(f"Epoch {epoch+1} — NPO: {epoch_npo_loss/n:.4f}, CE: {epoch_retain_loss/n:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="NPO Unlearning")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1, help="NPO inverse temperature")
    parser.add_argument("--gamma", type=float, default=1.0, help="Retain loss weight")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints/npo")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    print("=" * 60)
    print("  Negative Preference Optimization (NPO) Unlearning")
    print("  Ref: Zhang et al. 2024")
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

    # --- Load model ---
    print(f"\nLoading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    device = next(model.parameters()).device
    print_vram()

    # --- Tokenize ---
    forget_tok = forget_ds.map(lambda x: tokenize_qa(x, tokenizer, args.max_length),
                               batched=True, remove_columns=forget_ds.column_names)
    forget_tok.set_format("torch")
    retain_tok = retain_ds.map(lambda x: tokenize_qa(x, tokenizer, args.max_length),
                               batched=True, remove_columns=retain_ds.column_names)
    retain_tok.set_format("torch")

    forget_loader = DataLoader(forget_tok, batch_size=1, shuffle=False)
    retain_loader = DataLoader(retain_tok, batch_size=1, shuffle=True)
    forget_eval_loader = DataLoader(forget_tok, batch_size=1, shuffle=False)

    # --- Baseline perplexity ---
    print("\nComputing BASELINE forget-set perplexity...")
    ppl_before = compute_perplexity(model, forget_eval_loader, device)
    print(f"Forget perplexity BEFORE unlearning: {ppl_before:.2f}")

    # --- Precompute reference log-probs ---
    ref_log_probs = precompute_reference_log_probs(model, forget_loader, device)
    print(f"Cached {len(ref_log_probs)} reference log-prob values")

    # --- Train ---
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    forget_train_loader = DataLoader(forget_tok, batch_size=1, shuffle=True)

    print(f"\nStarting NPO training ({args.epochs} epochs, beta={args.beta}, gamma={args.gamma})...")
    print_vram()
    model = train_npo(
        model, forget_train_loader, retain_loader, ref_log_probs,
        optimizer, device, args.epochs, args.beta, args.gamma, args.grad_accum_steps
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
        "method": "npo",
        "model": args.model_name,
        "epochs": args.epochs,
        "lr": args.lr,
        "beta": args.beta,
        "gamma": args.gamma,
        "forget_samples": len(forget_ds),
        "ppl_before": ppl_before,
        "ppl_after": ppl_after,
        "ppl_ratio": ppl_after / max(ppl_before, 1e-8),
        "timestamp": timestamp,
        "dry_run": args.dry_run,
    }
    results_path = os.path.join(results_dir, f"npo_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    print("\n" + "=" * 60)
    print(f"  RESULT: Forget perplexity {ppl_before:.2f} -> {ppl_after:.2f} ({ppl_after/max(ppl_before,1e-8):.2f}x)")
    print("=" * 60)


if __name__ == "__main__":
    main()
