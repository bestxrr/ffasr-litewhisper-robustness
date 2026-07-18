#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/phase0_trainability.sh configs/experiments/phase0.yaml
