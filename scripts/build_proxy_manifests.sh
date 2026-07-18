#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
python -m src.evaluation.proxy_builder "${1:-configs/eval/proxy.yaml}"
