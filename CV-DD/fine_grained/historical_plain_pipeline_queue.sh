#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-CUB_imsize224}"
RECOVERY_SEED="${2:-41}"
IPC="${3:-1}"
STUDENT_SEED="${4:-42}"
[[ "$DATASET" == "CUB_imsize224" ]] || {
    echo "The deleted official launcher only documents CUB_imsize224" >&2
    exit 2
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_EXP_ROOT="${BASE_EXP_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1}"
DIAGNOSTIC_ROOT="$BASE_EXP_ROOT/diagnostics/historical_plain"
TEACHER_DIR="$DIAGNOSTIC_ROOT/teacher/$DATASET/tseed42"
PIPELINE_ROOT="$DIAGNOSTIC_ROOT/pipeline"
PREPARED_DATA_ROOT="$BASE_EXP_ROOT/datasets"
WAIT_FOR_STATUS="${WAIT_FOR_STATUS:-$BASE_EXP_ROOT/queue_status/CUB_imsize224_rseed43.status}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
LOG_ROOT="$DIAGNOSTIC_ROOT/logs/jobs"
STATUS="$DIAGNOSTIC_ROOT/status_${DATASET}_rseed${RECOVERY_SEED}_ipc${IPC}_sseed${STUDENT_SEED}.txt"
mkdir -p "$LOG_ROOT" "$TEACHER_DIR"

while [[ ! -f "$WAIT_FOR_STATUS" ]] || ! grep -q 'sampling complete' "$WAIT_FOR_STATUS"; do
    echo "$(date --iso-8601=seconds) waiting for status: $WAIT_FOR_STATUS"
    sleep "$WAIT_SECONDS"
done

echo "$(date --iso-8601=seconds) historical plain teacher started" > "$STATUS"
python -u "$ROOT_DIR/fine_grained/train_historical_plain_teacher.py" \
    --dataset-name "$DATASET" --data-dir "$PREPARED_DATA_ROOT/$DATASET" \
    --output-dir "$TEACHER_DIR" --seed 42 --epochs 100 --batch-size 32 \
    --lr 1e-2 --momentum 0.9 --weight-decay 1e-4 --eta-min 1e-5 \
    --skip-completed > "$LOG_ROOT/teacher_${DATASET}_tseed42.log" 2>&1
if ! python "$ROOT_DIR/fine_grained/check_teacher_gate.py" \
    --dataset-name "$DATASET" --teacher-dir "$TEACHER_DIR" \
    > "$LOG_ROOT/teacher_gate_${DATASET}_tseed42.log" 2>&1; then
    echo "$(date --iso-8601=seconds) CAL-reference gate failed; continuing exact historical arm" \
        >> "$STATUS"
fi

export EXP_ROOT="$PIPELINE_ROOT"
export PREPARED_DATA_ROOT
export TEACHER_DIR_OVERRIDE="$TEACHER_DIR"
export TEACHER_SEED=42
export PATCH_SEED=42

echo "$(date --iso-8601=seconds) historical plain pipeline started" > "$STATUS"
bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" patches "$DATASET" 42 \
    > "$LOG_ROOT/patches_${DATASET}.log" 2>&1
bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" recover "$DATASET" "$RECOVERY_SEED" \
    > "$LOG_ROOT/recover_${DATASET}_rseed${RECOVERY_SEED}.log" 2>&1
bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" sample "$DATASET" "$RECOVERY_SEED" \
    > "$LOG_ROOT/sample_${DATASET}_rseed${RECOVERY_SEED}.log" 2>&1
bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" relabel "$DATASET" "$RECOVERY_SEED" "$IPC" \
    > "$LOG_ROOT/relabel_${DATASET}_rseed${RECOVERY_SEED}_ipc${IPC}.log" 2>&1
bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" eval \
    "$DATASET" "$RECOVERY_SEED" "$IPC" "$STUDENT_SEED" \
    > "$LOG_ROOT/eval_${DATASET}_rseed${RECOVERY_SEED}_ipc${IPC}_sseed${STUDENT_SEED}.log" 2>&1
echo "$(date --iso-8601=seconds) historical plain pipeline complete" > "$STATUS"
