#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

# Fixed budgets estimated from the downloaded Lambda timings, not a timed stop.
echo "Learning curve | both tiny planners, Cartpole | estimated total ~4 hours (~2 offline + ~2 online)"
echo "Budget/model | offline=24000 updates | online=20000 updates / 157952 environment steps | serial models"
exec bash "$REPO_DIR/scripts/run_physical_controller_check.sh" \
  --eval-updates 1000 6000 12000 18000 24000 \
  --online-schedule-multiplier 2 --online-steps 157952 \
  --online-eval-steps 4096 41024 80000 118976 \
  --save-checkpoints \
  --output "runs/physical_learning_curve_$(date -u +%Y%m%d_%H%M%S)" \
  "$@"
