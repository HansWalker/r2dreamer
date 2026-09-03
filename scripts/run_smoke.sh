#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_DIR=${ENV_DIR:-"$REPO_DIR/../environment"}
PYTHON=${PYTHON:-"$ENV_DIR/bin/python"}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}

export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

cd "$REPO_DIR"
exec "$PYTHON" -u main.py --config-name dmc_smoke "$@"
