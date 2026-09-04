#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_DIR=${ENV_DIR:-"$REPO_DIR/../environment"}
PYTHON=${PYTHON:-"$ENV_DIR/bin/python"}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
PREFLIGHT_ROOT=${PREFLIGHT_ROOT:-"/tmp/r2dreamer_preflight_$(date +%Y%m%d_%H%M%S)"}

if [[ -z ${TDMPC2_DIR:-} ]]; then
    echo "Set TDMPC2_DIR to the local TD-MPC2 checkout before running the preflight." >&2
    exit 2
fi
export CUDA_HOME
export TDMPC2_DIR
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export DMC_SMOKE_DATA_DIR="$PREFLIGHT_ROOT/data"

RUN_DIR="$PREFLIGHT_ROOT/runs"
RUN_OVERRIDES=()
if [[ -d $PREFLIGHT_ROOT ]]; then
    echo "Preflight | resuming=$PREFLIGHT_ROOT"
    RUN_OVERRIDES=(
        --override stages.collect=false
        --override training.resume=true
        --override training.overwrite=false
    )
fi
mkdir -p "$PREFLIGHT_ROOT"
cd "$REPO_DIR"

echo "Preflight | dependency and GPU runtime checks"
"$PYTHON" -m scripts.check_dmc_setup

echo "Preflight | collection, training, and evaluation"
"$PYTHON" -u main.py \
    --config-name dmc_preflight \
    --override "output_dir=$RUN_DIR" \
    "${RUN_OVERRIDES[@]}"

"$PYTHON" - "$RUN_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
files = sorted(root.glob("*/*/*/seed_*/evaluation.json"))
expected = 13
errors = []

if len(files) != expected:
    errors.append(f"expected {expected} evaluations, found {len(files)}")

online_families = {"dreamer", "storm", "tdmpc2"}
for path in files:
    result = json.loads(path.read_text(encoding="utf-8"))
    name = "/".join(path.relative_to(root).parts[:-1])
    prediction = result["physical_state_prediction"]
    values = {
        "return": float(result["mean_return"]),
        "success": float(result["task_success_rate"]),
        "nrmse": float(prediction["mean_nrmse"]),
    }
    family = result["model_family"]
    expected_phase = "online" if family in online_families else "expert"

    protocol = result.get("experiment_protocol")
    if not protocol or protocol == "legacy_unversioned":
        errors.append(f"{name}: checkpoint protocol is missing")
    elif protocol != result.get("evaluation_protocol"):
        errors.append(f"{name}: training and evaluation protocols differ")
    if result.get("dataset_role") != "held_out":
        errors.append(f"{name}: evaluation dataset is not held out")
    if result.get("checkpoint_phase") != expected_phase:
        errors.append(f"{name}: expected {expected_phase} checkpoint")
    if int(result.get("expert_updates", 0)) < 2:
        errors.append(f"{name}: fewer than two expert updates")
    if family in online_families and int(result.get("environment_steps", 0)) < 8:
        errors.append(f"{name}: online phase did not reach eight steps")
    if not Path(result["checkpoint"]).is_file():
        errors.append(f"{name}: checkpoint is missing")
    if not all(math.isfinite(value) for value in values.values()):
        errors.append(f"{name}: non-finite evaluation metric")

    print(
        f"{name:55} "
        f"phase={expected_phase:6} "
        f"return={values['return']:8.2f} "
        f"success={100 * values['success']:6.1f}% "
        f"nrmse={values['nrmse']:.3f}"
    )

for path in root.rglob("*.log"):
    text = path.read_text(errors="replace")
    latest_attempt = text.rsplit("\n$ ", 1)[-1]
    if "Traceback (most recent call last):" in latest_attempt:
        errors.append(f"traceback in {path}")

if errors:
    print("\nPREFLIGHT FAILED:")
    print("\n".join(f"- {error}" for error in errors))
    raise SystemExit(1)

print(f"\nPREFLIGHT PASSED: all {expected} model runs completed successfully.")
PY

echo "Preflight | output=$PREFLIGHT_ROOT"
