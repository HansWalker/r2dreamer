#!/usr/bin/env bash
# Same two-model Cartpole work as the previous ~4h46m run, plus a small ranking loss.
set -euo pipefail
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$REPO_DIR/scripts/run_forecast_online.sh" \
  --task cartpole_balance_sparse --training-horizon 5 --offline-updates 2000 \
  --retain-offline --goal-ranking-weight 0.1 --goal-ranking-margin 0.1 \
  --goal-ranking-pairs 32 "$@"
