#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_EXP_ROOT="${BASE_EXP_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1}"
DIAGNOSTIC_ROOT="$BASE_EXP_ROOT/diagnostics/historical_intended"
TEACHER_DIR="$DIAGNOSTIC_ROOT/teacher/CUB_imsize224/tseed42"
TEACHER_STATUS="$DIAGNOSTIC_ROOT/status_CUB_imsize224_tseed42.txt"
PIPELINE_ROOT="$DIAGNOSTIC_ROOT/pipeline"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
LOG_ROOT="$DIAGNOSTIC_ROOT/logs/jobs"
STATUS="$DIAGNOSTIC_ROOT/status_pipeline_CUB_r41_ipc1_s42.txt"
mkdir -p "$LOG_ROOT"

while [[ ! -f "$TEACHER_STATUS" ]] || ! grep -q 'teacher complete' "$TEACHER_STATUS"; do
    echo "$(date --iso-8601=seconds) waiting for intended teacher: $TEACHER_STATUS"
    sleep "$WAIT_SECONDS"
done

python - "$TEACHER_DIR/teacher_gate.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
if payload.get("passed") is not True:
    raise SystemExit(f"Intended historical teacher gate failed: {path}")
PY

export EXP_ROOT="$PIPELINE_ROOT"
export PREPARED_DATA_ROOT="$BASE_EXP_ROOT/datasets"
export TEACHER_DIR_OVERRIDE="$TEACHER_DIR"
export TEACHER_SEED=42
export PATCH_SEED=42

echo "$(date --iso-8601=seconds) intended historical pipeline started" > "$STATUS"
bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" patches CUB_imsize224 42 \
    > "$LOG_ROOT/patches_CUB_imsize224.log" 2>&1
bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" recover CUB_imsize224 41 \
    > "$LOG_ROOT/recover_CUB_imsize224_rseed41.log" 2>&1
bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" sample CUB_imsize224 41 \
    > "$LOG_ROOT/sample_CUB_imsize224_rseed41.log" 2>&1
bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" relabel CUB_imsize224 41 1 \
    > "$LOG_ROOT/relabel_CUB_imsize224_rseed41_ipc1.log" 2>&1
bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" eval CUB_imsize224 41 1 42 \
    > "$LOG_ROOT/eval_CUB_imsize224_rseed41_ipc1_sseed42.log" 2>&1
echo "$(date --iso-8601=seconds) intended historical pipeline complete" > "$STATUS"
