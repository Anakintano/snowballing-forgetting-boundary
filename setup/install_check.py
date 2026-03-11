"""
Dependency checker for Snowballing at the Forgetting Boundary project.
Run: python setup/install_check.py
"""
import importlib
import importlib.util
import sys
import os

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REQUIRED = {
    "torch": "PyTorch (deep learning framework)",
    "transformers": "HuggingFace Transformers (model loading)",
    "bitsandbytes": "BitsAndBytes (4-bit quantization)",
    "peft": "PEFT (LoRA/QLoRA fine-tuning)",
    "accelerate": "Accelerate (mixed precision + device mapping)",
    "datasets": "HuggingFace Datasets (TOFU benchmark)",
    "tqdm": "tqdm (progress bars)",
    "numpy": "NumPy",
    "sklearn": "scikit-learn (probing classifiers)",
}

OPTIONAL = {
    "baukit": "baukit (activation hooks for mechanistic interp)",
    "einops": "einops (tensor reshaping)",
    "matplotlib": "matplotlib (plotting)",
    "scipy": "scipy (statistical tests)",
}

def check(libs, label):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    missing = []
    for mod, desc in libs.items():
        spec = importlib.util.find_spec(mod)
        if spec is not None:
            try:
                m = importlib.import_module(mod)
                ver = getattr(m, "__version__", "unknown")
                print(f"  [OK]   {mod:20s} {ver:>12s}  -- {desc}")
            except Exception as e:
                print(f"  [WARN] {mod:20s} {'import err':>12s}  -- {e}")
                missing.append(mod)
        else:
            print(f"  [MISS] {mod:20s} {'MISSING':>12s}  -- {desc}")
            missing.append(mod)
    return missing

def check_gpu():
    print(f"\n{'='*50}")
    print(f"  GPU STATUS")
    print(f"{'='*50}")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
            print(f"  GPU: {name}")
            print(f"  VRAM total: {total:.2f} GB")
            print(f"  VRAM free:  {free:.2f} GB")
            if total < 4.5:
                print(f"  [WARN] Low VRAM -- use 4-bit quantization for all models")
        else:
            print("  [MISS] CUDA not available")
    except ImportError:
        print("  [MISS] PyTorch not installed -- cannot check GPU")

if __name__ == "__main__":
    print("Snowballing at the Forgetting Boundary -- Dependency Check")
    print(f"Python: {sys.version}")

    missing_req = check(REQUIRED, "REQUIRED LIBRARIES")
    missing_opt = check(OPTIONAL, "OPTIONAL LIBRARIES")
    check_gpu()

    print(f"\n{'='*50}")
    if missing_req:
        print(f"  [WARN] Missing {len(missing_req)} required: {', '.join(missing_req)}")
        print(f"  Install with:")
        print(f"    pip install {' '.join(missing_req)}")
    else:
        print(f"  [OK] All required libraries installed!")
    if missing_opt:
        print(f"  [INFO] Missing {len(missing_opt)} optional: {', '.join(missing_opt)}")
    print(f"{'='*50}")
