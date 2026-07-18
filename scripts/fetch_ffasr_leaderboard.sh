#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
python - <<'PY'
from src.evaluation.ffasr_api import write_leaderboard_snapshot
summary = write_leaderboard_snapshot("artifacts/leaderboard")
print(summary)
PY
