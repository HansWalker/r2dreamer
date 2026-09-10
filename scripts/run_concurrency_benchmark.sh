#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUTPUT=runs/concurrency_bf16_check
ONLY=all
FORWARD=()
while (($#)); do
  case "$1" in
    --output) OUTPUT=${2:?--output requires a directory}; shift 2 ;;
    --only) ONLY=${2:?--only requires dreamer, tdmpc2, pairs, or all}; shift 2 ;;
    --help|-h)
      printf '%s\n' \
        'Usage: bash scripts/run_concurrency_benchmark.sh --dataset-root DIR [--only dreamer|tdmpc2|pairs|all] [--output DIR] [--dry-run]' \
        'Other options are forwarded to smoke_training (e.g. --updates or --device).' \
        'Default: 21 workers including 9 shared serial references; no checkpoints or evaluations.'
      exit 0 ;;
    *) FORWARD+=("$1"); shift ;;
  esac
done
case "$ONLY" in
  all|dreamer|tdmpc2|pairs) ;;
  *) printf 'Unknown --only section: %s\n' "$ONLY" >&2; exit 2 ;;
esac

STATUS=0
for SECTION in dreamer tdmpc2 pairs; do
  if [[ "$ONLY" != all && "$ONLY" != "$SECTION" ]]; then
    continue
  fi
  case "$SECTION" in
    dreamer)
      OPTIONS=(--scenarios cartpole_balance_sparse reacher ball_in_cup
        --models dreamer/mamba3 --scenario-workers 3 --rollout-steps 64) ;;
    tdmpc2)
      OPTIONS=(--scenarios cartpole_balance_sparse reacher ball_in_cup
        --models tdmpc2/default --scenario-workers 2) ;;
    pairs)
      OPTIONS=(--scenarios ball_in_cup
        --pair temporal_straightening/default leworldmodel/default
        --pair temporal_straightening/default storm/mamba3
        --pair leworldmodel/default storm/mamba3) ;;
  esac
  printf '\nConcurrency | section=%s | output=%s/%s\n' "$SECTION" "$OUTPUT" "$SECTION"
  if ! bash "$SCRIPT_DIR/run_training_smoke.sh" \
    --updates 32 --warmup-updates 8 --rollout-steps 8 --online-burst 4 --eval-steps 0 \
    "${OPTIONS[@]}" "${FORWARD[@]}" --output "$OUTPUT/$SECTION"; then
    STATUS=1
  fi
done
exit "$STATUS"
