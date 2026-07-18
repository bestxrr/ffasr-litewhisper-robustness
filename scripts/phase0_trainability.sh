#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
export TORCH_HOME="$PWD/.cache/torch"
export WANDB_DIR="$PWD/artifacts/wandb"
python -m src.training.phase0_trainability "${1:-configs/experiments/phase0.yaml}"
