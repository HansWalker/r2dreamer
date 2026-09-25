#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_DIR=${ENV_DIR:-"$REPO_DIR/../environment"}
PYTHON=${PYTHON:-"$ENV_DIR/bin/python"}
cd "$REPO_DIR"
export MUJOCO_GL=${MUJOCO_GL:-egl}
exec "$PYTHON" -u -m scripts.evaluate_history_context "$@"
