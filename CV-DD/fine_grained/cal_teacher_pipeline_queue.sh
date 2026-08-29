#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: cal_teacher_pipeline_queue.sh DATASET RECOVERY_SEED IPC STUDENT_SEED}"
RECOVERY_SEED="${2:?missing recovery seed}"
IPC="${3:?missing IPC}"
STUDENT_SEED="${4:?missing student seed}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_EXP_ROOT="${BASE_EXP_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1}"
CAL_ROOT="$BASE_EXP_ROOT/diagnostics/cal_teacher/$DATASET"
TEACHER_DIR="$CAL_ROOT/extracted"
CAL_STATUS="$CAL_ROOT/status.txt"
PIPELINE_ROOT="$BASE_EXP_ROOT/diagnostics/cal_teacher_pipeline"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
LOG_ROOT="$PIPELINE_ROOT/logs/jobs"
STATUS="$PIPELINE_ROOT/status_${DATASET}_rseed${RECOVERY_SEED}_ipc${IPC}_sseed${STUDENT_SEED}.txt"
mkdir -p "$LOG_ROOT"

while [[ ! -f "$CAL_STATUS" ]] || ! grep -q 'diagnostic complete' "$CAL_STATUS"; do
    echo "$(date --iso-8601=seconds) waiting for CAL teacher: $CAL_STATUS"
    sleep "$WAIT_SECONDS"
done

python - "$TEACHER_DIR/teacher_gate.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("passed") is not True:
    raise SystemExit(f"CAL backbone teacher gate did not pass: {path}")
PY

export EXP_ROOT="$PIPELINE_ROOT"
export PREPARED_DATA_ROOT="$BASE_EXP_ROOT/datasets"
export TEACHER_DIR_OVERRIDE="$TEACHER_DIR"
export TEACHER_SEED=42
export PATCH_SEED=42

echo "$(date --iso-8601=seconds) CAL-backbone pipeline started" > "$STATUS"
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
echo "$(date --iso-8601=seconds) CAL-backbone pipeline complete" > "$STATUS"
