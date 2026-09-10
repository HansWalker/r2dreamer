#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/run_training_smoke.sh" \
  --scenarios ball_in_cup \
  --models temporal_straightening/default leworldmodel/default \
  --updates 12 --warmup-updates 4 --rollout-steps 4 --online-burst 4 \
  --eval-steps 3 --eval-batches 5 50 \
  --compare-gradient-batch 128 256 \
  --output runs/planner_check "$@"
