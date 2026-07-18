#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
export TORCH_HOME="$PWD/.cache/torch"
export WANDB_DIR="$PWD/artifacts/wandb"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$WANDB_DIR"
uv pip install -r requirements.txt
