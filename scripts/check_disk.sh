#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /venv/main/bin/activate
python - <<'PY'
from src.utils.disk import project_usage, assert_within_budget
usage = project_usage(".")
for key, value in usage.items():
    print(f"{key}\t{value:.3f} GB")
assert_within_budget(".", 48.0)
PY
