#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_DIR=${ENV_DIR:-"$REPO_DIR/../environment"}
PYTHON=${PYTHON:-"$ENV_DIR/bin/python"}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}

if [[ ! -x $PYTHON ]]; then
    echo "Python environment not found at $PYTHON. Run scripts/setup_dmc.sh first." >&2
    exit 2
fi
if [[ ! -x $CUDA_HOME/bin/nvcc ]]; then
    echo "CUDA toolkit not found at $CUDA_HOME. Set CUDA_HOME to the directory containing bin/nvcc." >&2
    exit 2
fi
if [[ -z ${TDMPC2_DIR:-} || ! -f $TDMPC2_DIR/tdmpc2/config.yaml ]]; then
    echo "Set TDMPC2_DIR to a TD-MPC2 checkout containing tdmpc2/config.yaml." >&2
    exit 2
fi

export CUDA_HOME
export TDMPC2_DIR
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

cd "$REPO_DIR"
exec "$PYTHON" -u main.py --config-name dmc_benchmark "$@"
