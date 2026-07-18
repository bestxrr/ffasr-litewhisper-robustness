#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
export HF_HOME="$PWD/.cache/huggingface"
export TORCH_HOME="$PWD/.cache/torch"
python -m src.data.materialize_parquet_audio "${1:-configs/data/librispeech_proxy_clean_parquet.yaml}"
