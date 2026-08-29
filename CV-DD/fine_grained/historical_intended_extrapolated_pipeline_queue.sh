#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: historical_intended_extrapolated_pipeline_queue.sh DATASET}"
case "$DATASET" in
    A_imsize224|SC_imsize224) ;;
    *) echo "Expected A_imsize224 or SC_imsize224" >&2; exit 2 ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_EXP_ROOT="${BASE_EXP_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1}"
DIAGNOSTIC_ROOT="$BASE_EXP_ROOT/diagnostics/historical_intended"
TEACHER_DIR="$DIAGNOSTIC_ROOT/teacher/$DATASET/tseed42"
TEACHER_COMPLETE="$TEACHER_DIR/complete.json"
PIPELINE_ROOT="$DIAGNOSTIC_ROOT/pipeline"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
LOG_ROOT="$DIAGNOSTIC_ROOT/logs/jobs"
STATUS="$DIAGNOSTIC_ROOT/status_pipeline_${DATASET}_r41_ipc1_s42.txt"
mkdir -p "$LOG_ROOT"

while [[ ! -f "$TEACHER_COMPLETE" ]]; do
    echo "$(date --iso-8601=seconds) waiting for intended teacher: $TEACHER_COMPLETE"
    sleep "$WAIT_SECONDS"
done

python "$ROOT_DIR/fine_grained/check_teacher_gate.py" \
    --dataset-name "$DATASET" --teacher-dir "$TEACHER_DIR" \
    > "$LOG_ROOT/teacher_gate_${DATASET}_tseed42.log" 2>&1

export EXP_ROOT="$PIPELINE_ROOT"
export PREPARED_DATA_ROOT="$BASE_EXP_ROOT/datasets"
export TEACHER_DIR_OVERRIDE="$TEACHER_DIR"
export TEACHER_SEED=42
export PATCH_SEED=42

echo "$(date --iso-8601=seconds) intended extrapolated pipeline started" > "$STATUS"
bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" patches "$DATASET" 42 \
    > "$LOG_ROOT/patches_${DATASET}.log" 2>&1
bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" recover "$DATASET" 41 \
    > "$LOG_ROOT/recover_${DATASET}_rseed41.log" 2>&1
bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" sample "$DATASET" 41 \
    > "$LOG_ROOT/sample_${DATASET}_rseed41.log" 2>&1
bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" relabel "$DATASET" 41 1 \
    > "$LOG_ROOT/relabel_${DATASET}_rseed41_ipc1.log" 2>&1
bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" eval "$DATASET" 41 1 42 \
    > "$LOG_ROOT/eval_${DATASET}_rseed41_ipc1_sseed42.log" 2>&1
echo "$(date --iso-8601=seconds) intended extrapolated pipeline complete" > "$STATUS"
