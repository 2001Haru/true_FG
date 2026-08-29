#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: student_initialization_queue.sh DATASET RECOVERY_SEED IPC STUDENT_SEED}"
RECOVERY_SEED="${2:?missing recovery seed}"
IPC="${3:?missing IPC}"
STUDENT_SEED="${4:?missing student seed}"
BASE_EXP_ROOT="${BASE_EXP_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1}"
PIPELINE_ROOT="$BASE_EXP_ROOT/diagnostics/historical_intended/pipeline"
PREREQUISITE_RESULT="${WAIT_FOR_RESULT:-$PIPELINE_ROOT/results/$DATASET/rseed${RECOVERY_SEED}/ipc${IPC}_sseed${STUDENT_SEED}.json}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIAGNOSTIC_ROOT="$BASE_EXP_ROOT/diagnostics/student_imagenet"
LOG_ROOT="$DIAGNOSTIC_ROOT/logs/jobs"
STATUS="$DIAGNOSTIC_ROOT/status_${DATASET}_rseed${RECOVERY_SEED}_ipc${IPC}_sseed${STUDENT_SEED}.txt"
mkdir -p "$LOG_ROOT"

while [[ ! -f "$PREREQUISITE_RESULT" ]]; do
    echo "$(date --iso-8601=seconds) waiting for random-student result: $PREREQUISITE_RESULT"
    sleep "$WAIT_SECONDS"
done

export EXP_ROOT="$PIPELINE_ROOT"
export PREPARED_DATA_ROOT="$BASE_EXP_ROOT/datasets"
export RESULT_ROOT="$DIAGNOSTIC_ROOT/results"
export POST_EVAL_ROOT="$DIAGNOSTIC_ROOT/post_eval"
export STUDENT_INITIALIZATION=imagenet-v1
export STUDENT_TEMPERATURE=20

echo "$(date --iso-8601=seconds) ImageNet student started" > "$STATUS"
bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" eval \
    "$DATASET" "$RECOVERY_SEED" "$IPC" "$STUDENT_SEED" \
    > "$LOG_ROOT/eval_${DATASET}_r${RECOVERY_SEED}_ipc${IPC}_s${STUDENT_SEED}.log" 2>&1
echo "$(date --iso-8601=seconds) ImageNet student complete" > "$STATUS"
