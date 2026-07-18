#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
python - <<'PY'
import os, shutil, subprocess, sys
print("python", sys.version.split()[0])
for key in ["HF_TOKEN","HUGGING_FACE_HUB_TOKEN","WANDB_API_KEY","CONTAINER_API_KEY"]:
    print(f"{key}=" + ("present" if os.environ.get(key) else "absent"))
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
except Exception as exc:
    print("torch_error", repr(exc))
for mod in ["transformers","safetensors","peft","soundfile","librosa","jiwer","datasets"]:
    try:
        m=__import__(mod)
        print(mod, getattr(m, "__version__", "ok"))
    except Exception as exc:
        print(mod, "missing", type(exc).__name__)
print("gh=" + ("present" if shutil.which("gh") else "absent"))
PY
nvidia-smi
bash scripts/check_disk.sh
