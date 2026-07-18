#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <errors.csv> <output-dir>" >&2
  exit 2
fi

cd "$(dirname "$0")/.."
source /venv/main/bin/activate
python -m src.analysis.errors --input "$1" --output-dir "$2"
