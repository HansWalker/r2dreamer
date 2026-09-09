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

echo "Preflight | production parameter budgets"
"$PYTHON" -m scripts.model_size_report

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

from training.protocol import EVALUATION_PROTOCOL

root = Path(sys.argv[1]).resolve()
files = sorted(root.glob("*/*/*/seed_*/evaluation.json"))
expected = 13
errors = []

if len(files) != expected:
    errors.append(f"expected {expected} evaluations, found {len(files)}")

for path in files:
    result = json.loads(path.read_text(encoding="utf-8"))
    name = "/".join(path.relative_to(root).parts[:-1])
    prediction = result["physical_state_prediction"]
    values = {
        "return": float(result["mean_return"]),
        "success": float(result["task_success_rate"]),
        "sustained": float(result["sustained_success_rate"]),
    }
    protocol = result.get("experiment_protocol")
    if not protocol or protocol == "legacy_unversioned":
        errors.append(f"{name}: checkpoint protocol is missing")
    elif result.get("evaluation_protocol") != EVALUATION_PROTOCOL:
        errors.append(f"{name}: wrong or missing evaluation protocol")
    if result.get("dataset_role") != "held_out":
        errors.append(f"{name}: evaluation dataset is not held out")
    if result.get("checkpoint_phase") != "online":
        errors.append(f"{name}: expected online checkpoint")
    if int(result.get("expert_updates", 0)) < 2:
        errors.append(f"{name}: fewer than two expert updates")
    if int(result.get("expert_sampled_observations", 0)) != 8:
        errors.append(f"{name}: expert sampled-observation count is incorrect")
    if int(result.get("environment_steps", 0)) < 8:
        errors.append(f"{name}: online phase did not reach eight steps")
    if int(result.get("online_updates", 0)) < 1:
        errors.append(f"{name}: online phase performed no optimizer update")
    if int(result.get("online_world_model_observations", 0)) != 4:
        errors.append(f"{name}: online sampled-observation count is incorrect")
    if not Path(result["checkpoint"]).is_file():
        errors.append(f"{name}: checkpoint is missing")
    if not all(math.isfinite(value) for value in values.values()):
        errors.append(f"{name}: non-finite evaluation metric")
    coordinates = prediction["state_coordinates"]
    rmse = prediction["rmse"]
    expected_head_updates = int(result["expert_updates"]) + int(result["online_updates"])
    if prediction.get("readout_updates") != expected_head_updates or prediction.get("readout_examples") != expected_head_updates:
        errors.append(f"{name}: physical-state head did not train once per model update")
    if prediction.get("evaluation_fitting") is not False:
        errors.append(f"{name}: evaluation did not use the fixed checkpoint head")
    if not coordinates or set(rmse) != {"1"}:
        errors.append(f"{name}: missing physical-state prediction metrics")
    for horizon, state_errors in rmse.items():
        if set(state_errors) != set(coordinates):
            errors.append(f"{name}: incomplete RMSE coordinates at horizon {horizon}")
        if not all(math.isfinite(float(value)) and float(value) >= 0 for value in state_errors.values()):
            errors.append(f"{name}: invalid RMSE at horizon {horizon}")

    print(
        f"{name:55} "
        "phase=online "
        f"return={values['return']:8.2f} "
        f"success={100 * values['success']:6.1f}% "
        f"sustained={100 * values['sustained']:6.1f}%"
    )
    for horizon, state_errors in rmse.items():
        details = ", ".join(f"{key}={value:.4g}" for key, value in state_errors.items())
        print(f"  RMSE (original units) | horizon={horizon} | {details}")

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
