#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_DIR=${ENV_DIR:-"$REPO_DIR/../environment"}
PYTHON=${PYTHON:-python3}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
MAMBA_COMMIT=a14b1dff0454a3bc27d9eb31355dc01e4b2490ec

cd "$REPO_DIR"

"$PYTHON" - <<'PY'
import sys

if sys.version_info[:2] not in {(3, 10), (3, 11)}:
    raise SystemExit(f"DMC experiments require Python 3.10 or 3.11, got {sys.version.split()[0]}")
PY

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "CUDA 12.8 toolkit not found at $CUDA_HOME." >&2
    echo "Set CUDA_HOME to the directory containing bin/nvcc." >&2
    exit 1
fi

if ! "$PYTHON" - <<'PY'
import ctypes.util
import sys

sys.exit(0 if ctypes.util.find_library("z3") else 1)
PY
then
    echo "libz3.so is missing. On Ubuntu, run: sudo apt-get install -y libz3-dev" >&2
    exit 1
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    "$PYTHON" -m venv "$ENV_DIR"
fi

PY="$ENV_DIR/bin/python"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$PY" -m pip install --upgrade \
    "pip>=25,<26" \
    "setuptools==77.0.3" \
    "wheel>=0.45,<0.46" \
    "ninja>=1.11,<2" \
    "packaging>=24,<26"

"$PY" -m pip install \
    "torch==2.8.0" \
    --index-url https://download.pytorch.org/whl/cu128

"$PY" - <<'PY'
import os
import re
import subprocess

import torch

nvcc = subprocess.check_output([os.environ["CUDA_HOME"] + "/bin/nvcc", "--version"], text=True)
match = re.search(r"release (\d+\.\d+)", nvcc)
toolkit = match.group(1) if match else "unknown"
if torch.version.cuda != toolkit:
    raise SystemExit(f"PyTorch uses CUDA {torch.version.cuda}, but nvcc is CUDA {toolkit}")
PY

"$PY" -m pip install -e ".[dmc]" "pytest>=8,<9" "pre-commit>=4,<5"
QUACK_REQUIREMENT=$(grep '^quack-kernels==' requirements/mamba3-cu128.txt)
TRITON_REQUIREMENT=$(grep '^triton==' requirements/mamba3-cu128.txt)
"$PY" -m pip install -r <(grep -Ev '^(quack-kernels|triton)==' requirements/mamba3-cu128.txt)
# Quack 0.5.3 declares an unreleased CUTLASS 4.6 dependency, but this Mamba
# commit uses its helper API with the tested CUTLASS 4.5.2 runtime above.
# Torch 2.8 pins Triton 3.4, but Mamba3 requires the small matrix support added
# in Triton 3.5. Neither override changes Torch's compiled CUDA extensions.
"$PY" -m pip install --no-deps "$QUACK_REQUIREMENT" "$TRITON_REQUIREMENT"

MAMBA_FORCE_BUILD=TRUE MAMBA_SKIP_CUDA_BUILD=FALSE \
    "$PY" -m pip install \
    --no-cache-dir \
    --force-reinstall \
    --no-deps \
    --no-build-isolation \
    "git+https://github.com/state-spaces/mamba.git@$MAMBA_COMMIT"

"$PY" -m scripts.check_dmc_setup

echo
echo "DMC environment ready."
echo "Activate it with: source $ENV_DIR/bin/activate"
echo "Then verify every configured model with: python -m scripts.smoke_models"
