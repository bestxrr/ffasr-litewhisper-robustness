#!/usr/bin/env bash
set -euo pipefail
echo "Official FFASR baseline reproduction requires submitting/evaluating through the Space or maintainer job path."
echo "This script records the current public row and protocol snapshot locally."
cd "$(dirname "$0")/.."
bash scripts/fetch_ffasr_leaderboard.sh
