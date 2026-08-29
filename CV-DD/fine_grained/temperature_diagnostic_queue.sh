#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: temperature_diagnostic_queue.sh DATASET RECOVERY_SEED IPC STUDENT_SEED TEMPERATURE}"
RECOVERY_SEED="${2:?missing recovery seed}"
IPC="${3:?missing IPC}"
STUDENT_SEED="${4:?missing student seed}"
TEMPERATURE="${5:?missing temperature}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1}"
FORMAL_RESULT="$EXP_ROOT/results/$DATASET/rseed${RECOVERY_SEED}/ipc${IPC}_sseed${STUDENT_SEED}.json"
TAG="temperature_${TEMPERATURE//./p}"
DIAGNOSTIC_ROOT="$EXP_ROOT/diagnostics/$TAG"
STATUS="$DIAGNOSTIC_ROOT/status_${DATASET}_rseed${RECOVERY_SEED}_ipc${IPC}_sseed${STUDENT_SEED}.txt"
mkdir -p "$DIAGNOSTIC_ROOT"

while [[ ! -f "$FORMAL_RESULT" ]]; do
    echo "$(date --iso-8601=seconds) waiting for formal result: $FORMAL_RESULT"
    sleep 60
done

echo "$(date --iso-8601=seconds) formal result observed; launching T=$TEMPERATURE" > "$STATUS"
STUDENT_TEMPERATURE="$TEMPERATURE" \
RESULT_ROOT="$DIAGNOSTIC_ROOT/results" \
POST_EVAL_ROOT="$DIAGNOSTIC_ROOT/post_eval" \
bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" eval \
    "$DATASET" "$RECOVERY_SEED" "$IPC" "$STUDENT_SEED"
echo "$(date --iso-8601=seconds) diagnostic complete" > "$STATUS"
