#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_DIR=${ENV_DIR:-"$REPO_DIR/../environment"}
PYTHON=${PYTHON:-"$ENV_DIR/bin/python"}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$OMP_NUM_THREADS}
cd "$REPO_DIR"
exec "$PYTHON" -u -m scripts.train_paper_faithful_check "$@"
