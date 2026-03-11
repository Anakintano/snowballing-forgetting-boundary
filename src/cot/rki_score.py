"""
Residual Knowledge Interference (RKI) Score.

RKI(l, d) = I(h_l^(d); C_forget)

Measures mutual information between hidden states at layer l during CoT step d
and the forget-concept representation. Uses a probe-based MI estimator.

# RESEARCH NOTE: True MI is intractable in high dimensions. We use two proxies:
# 1. Linear probe accuracy (lower bound on MI via data processing inequality)
# 2. Cosine similarity to forget-set centroid (fast, interpretable)
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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


def extract_hidden_states(model, tokenizer, text, layers=None):
    """Extract hidden states from specified layers for a given text.

    # HARDWARE NOTE: Hidden states are moved to CPU immediately to save VRAM.
    # For Qwen2.5-1.5B with 28 layers, each hidden state is [seq_len, 1536].
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # outputs.hidden_states is a tuple of (n_layers+1) tensors
    hidden_states = outputs.hidden_states

    if layers is None:
        layers = list(range(len(hidden_states)))

    result = {}
    for l in layers:
        if l < len(hidden_states):
            # Mean pool over sequence length -> single vector per layer
            # HARDWARE NOTE: .cpu() immediately to free VRAM
            h = hidden_states[l][0].mean(dim=0).cpu().float().numpy()
            result[l] = h

    return result


def compute_forget_centroids(model, tokenizer, forget_ds, layers):
    """Compute centroid of forget-set hidden states per layer.

    # RESEARCH NOTE: C_forget is the mean activation vector of the forget set.
    # This represents the "concept" that should have been erased.
    # RKI measures how much a CoT hidden state still "points toward" this centroid.
    """
    print("Computing forget-set centroids...")
    all_hidden = {l: [] for l in layers}

    for sample in tqdm(forget_ds, desc="Forget centroids"):
        text = f"Question: {sample['question']}\nAnswer: {sample['answer']}"
        hs = extract_hidden_states(model, tokenizer, text, layers)
        for l in layers:
            if l in hs:
                all_hidden[l].append(hs[l])

    centroids = {}
    for l in layers:
        if all_hidden[l]:
            centroids[l] = np.mean(all_hidden[l], axis=0)
    return centroids


def compute_rki_cosine(hidden_state, centroid):
    """RKI proxy: cosine similarity between hidden state and forget centroid."""
    norm_h = np.linalg.norm(hidden_state)
    norm_c = np.linalg.norm(centroid)
    if norm_h < 1e-8 or norm_c < 1e-8:
        return 0.0
    return float(np.dot(hidden_state, centroid) / (norm_h * norm_c))


def compute_rki_probe(forget_hidden, retain_hidden):
    """RKI proxy: train a linear probe to distinguish forget vs retain activations.

    # RESEARCH NOTE: Probe accuracy is a lower bound on MI.
    # If a linear classifier can distinguish forget-set activations from retain-set
    # activations at a given layer, there is residual information about the forget concept.
    # Post-unlearning, we expect this accuracy to DECREASE if unlearning was effective,
    # but to REMAIN HIGH at certain layers if knowledge is only partially erased.
    """
    n_forget = len(forget_hidden)
    n_retain = min(len(retain_hidden), n_forget * 2)  # balance classes roughly

    X = np.vstack(forget_hidden[:n_forget] + retain_hidden[:n_retain])
    y = np.array([1] * n_forget + [0] * n_retain)

    # Shuffle
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]

    # Simple train/test split
    split = int(0.7 * len(y))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return 0.5  # degenerate case

    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return float(acc)


def main():
    parser = argparse.ArgumentParser(description="Compute RKI scores")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter_path", type=str, default="models/checkpoints/ga")
    parser.add_argument("--output_dir", type=str, default="results/phase2")
    parser.add_argument("--n_layers", type=int, default=8, help="Sample N evenly-spaced layers")
    parser.add_argument("--n_retain_samples", type=int, default=40)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--mode", type=str, default="unlearned", choices=["base", "unlearned", "both"],
                        help="Which model to evaluate. Use 'unlearned' or 'base' separately to avoid reload segfault on Windows.")
    args = parser.parse_args()

    print("=" * 60)
    print("  RKI Score Computation")
    print("=" * 60)
    print_vram()

    # Load dataset
    forget_ds = load_dataset("locuslab/TOFU", "forget01", split="train")
    retain_ds = load_dataset("locuslab/TOFU", "retain99", split="train")

    if args.dry_run:
        forget_ds = forget_ds.select(range(min(5, len(forget_ds))))
        retain_ds = retain_ds.select(range(min(10, len(retain_ds))))
        args.n_retain_samples = 10
        args.n_layers = 4

    retain_subset = retain_ds.select(range(min(args.n_retain_samples, len(retain_ds))))

    # Determine layers to probe
    # RESEARCH NOTE: We sample layers evenly to get a cross-section of the model.
    # Early layers = low-level features, middle = semantic, late = task-specific.

    # --- Load model based on mode ---
    run_unlearned = args.mode in ("unlearned", "both")
    run_base = args.mode in ("base", "both")

    if run_unlearned:
        adapter = args.adapter_path
    else:
        adapter = None

    print(f"\nLoading model (mode={args.mode})...")
    model, tokenizer = load_model(args.model_name, adapter if run_unlearned else None)
    print_vram()

    # Determine total layers
    dummy = tokenizer("test", return_tensors="pt")
    dummy = {k: v.to(model.device) for k, v in dummy.items()}
    with torch.no_grad():
        out = model(**dummy, output_hidden_states=True)
    n_total_layers = len(out.hidden_states)
    print(f"Model has {n_total_layers} hidden state layers")

    layers = list(np.linspace(0, n_total_layers - 1, args.n_layers, dtype=int))
    print(f"Probing layers: {layers}")

    # Compute forget centroids
    centroids = compute_forget_centroids(model, tokenizer, forget_ds, layers)

    # Extract hidden states for forget and retain sets
    print("\nExtracting forget-set hidden states (unlearned model)...")
    forget_hidden = {l: [] for l in layers}
    for sample in tqdm(forget_ds, desc="Forget"):
        text = f"Question: {sample['question']}\nAnswer: {sample['answer']}"
        hs = extract_hidden_states(model, tokenizer, text, layers)
        for l in layers:
            if l in hs:
                forget_hidden[l].append(hs[l])

    print("Extracting retain-set hidden states (unlearned model)...")
    retain_hidden = {l: [] for l in layers}
    for sample in tqdm(retain_subset, desc="Retain"):
        text = f"Question: {sample['question']}\nAnswer: {sample['answer']}"
        hs = extract_hidden_states(model, tokenizer, text, layers)
        for l in layers:
            if l in hs:
                retain_hidden[l].append(hs[l])

    # --- Compute RKI scores ---
    model_label = "unlearned" if run_unlearned else "base"
    print("\n" + "=" * 60)
    print(f"  RKI SCORES ({model_label} Model)")
    print("=" * 60)

    rki_results = {model_label: {}, "layers": [int(l) for l in layers]}

    for l in layers:
        # Cosine similarity to forget centroid
        cosine_scores = [compute_rki_cosine(h, centroids[l]) for h in forget_hidden[l]]
        cosine_retain = [compute_rki_cosine(h, centroids[l]) for h in retain_hidden[l]]

        # Probe accuracy (MI lower bound)
        probe_acc = compute_rki_probe(forget_hidden[l], retain_hidden[l])

        rki_results[model_label][int(l)] = {
            "probe_accuracy": probe_acc,
            "forget_cosine_mean": float(np.mean(cosine_scores)) if cosine_scores else None,
            "forget_cosine_std": float(np.std(cosine_scores)) if cosine_scores else None,
            "retain_cosine_mean": float(np.mean(cosine_retain)) if cosine_retain else None,
            "cosine_gap": float(np.mean(cosine_scores) - np.mean(cosine_retain))
                         if cosine_scores and cosine_retain else None,
        }
        print(f"  Layer {l:3d}: probe_acc={probe_acc:.3f}, "
              f"forget_cos={np.mean(cosine_scores):.3f}, "
              f"retain_cos={np.mean(cosine_retain):.3f}, "
              f"gap={np.mean(cosine_scores) - np.mean(cosine_retain):.3f}")

    # NOTE: To compare base vs unlearned, run this script twice with --mode base
    # and --mode unlearned, then compare the JSON outputs. This avoids the
    # reload segfault on Windows (device_map + del model + reload crashes).
    print(f"\nDone. Run with --mode {'base' if run_unlearned else 'unlearned'} to get comparison data.")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rki_results["timestamp"] = timestamp
    rki_results["model"] = args.model_name
    rki_results["mode"] = args.mode
    rki_results["dry_run"] = args.dry_run
    out_path = os.path.join(args.output_dir, f"rki_{args.mode}_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(rki_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
