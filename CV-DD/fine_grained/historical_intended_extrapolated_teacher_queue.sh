#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: historical_intended_extrapolated_teacher_queue.sh DATASET [SEED]}"
SEED="${2:-42}"
case "$DATASET" in
    A_imsize224|SC_imsize224) ;;
    *) echo "Expected A_imsize224 or SC_imsize224" >&2; exit 2 ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_EXP_ROOT="${BASE_EXP_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1}"
WAIT_FOR_RESULT="${WAIT_FOR_RESULT:-$BASE_EXP_ROOT/diagnostics/historical_plain/teacher/$DATASET/tseed${SEED}/complete.json}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
OUTPUT_DIR="$BASE_EXP_ROOT/diagnostics/historical_intended/teacher/$DATASET/tseed${SEED}"
LOG_ROOT="$BASE_EXP_ROOT/diagnostics/historical_intended/logs/jobs"
STATUS="$BASE_EXP_ROOT/diagnostics/historical_intended/status_${DATASET}_tseed${SEED}.txt"
mkdir -p "$OUTPUT_DIR" "$LOG_ROOT"

while [[ ! -f "$WAIT_FOR_RESULT" ]]; do
    echo "$(date --iso-8601=seconds) waiting for prerequisite: $WAIT_FOR_RESULT"
    sleep "$WAIT_SECONDS"
done

echo "$(date --iso-8601=seconds) intended ImageNet-init teacher started" > "$STATUS"
python -u "$ROOT_DIR/fine_grained/train_historical_plain_teacher.py" \
    --dataset-name "$DATASET" --data-dir "$BASE_EXP_ROOT/datasets/$DATASET" \
    --output-dir "$OUTPUT_DIR" --seed "$SEED" --epochs 100 --batch-size 32 \
    --lr 1e-2 --momentum 0.9 --weight-decay 1e-4 --eta-min 1e-5 \
    --initialization imagenet-v1 --skip-completed \
    > "$LOG_ROOT/teacher_${DATASET}_tseed${SEED}.log" 2>&1
python "$ROOT_DIR/fine_grained/check_teacher_gate.py" \
    --dataset-name "$DATASET" --teacher-dir "$OUTPUT_DIR" \
    > "$LOG_ROOT/teacher_gate_${DATASET}_tseed${SEED}.log" 2>&1
echo "$(date --iso-8601=seconds) intended ImageNet-init teacher complete" > "$STATUS"
