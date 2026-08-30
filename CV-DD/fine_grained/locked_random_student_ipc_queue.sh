#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: locked_random_student_ipc_queue.sh DATASET IPC}"
IPC="${2:?missing IPC}"
[[ "$IPC" =~ ^(1|3|5)$ ]] || { echo "IPC must be 1, 3, or 5" >&2; exit 2; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_EXP_ROOT="${BASE_EXP_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1}"
PIPELINE_ROOT="$BASE_EXP_ROOT/diagnostics/historical_intended/pipeline"
FAIR_ROOT="$BASE_EXP_ROOT/diagnostics/student_random_sre2l_fd2"
LOG_ROOT="$FAIR_ROOT/logs/jobs"
STATUS_ROOT="$FAIR_ROOT/status"
LOCK_ROOT="$FAIR_ROOT/locks"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$LOCK_ROOT"

case "$DATASET" in
    CUB_imsize224|A_imsize224|SC_imsize224) ;;
    *) echo "unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

command -v flock >/dev/null 2>&1 || { echo "flock is required" >&2; exit 1; }
exec 8>"$LOCK_ROOT/${DATASET}_r41_ipc${IPC}.lock"
flock -n 8 || { echo "random-student IPC queue already running: $DATASET IPC$IPC" >&2; exit 1; }

export EXP_ROOT="$PIPELINE_ROOT"
export PREPARED_DATA_ROOT="$BASE_EXP_ROOT/datasets"
export TEACHER_DIR_OVERRIDE="$BASE_EXP_ROOT/diagnostics/historical_intended/teacher/$DATASET/tseed42"
export TEACHER_SEED=42
export PATCH_SEED=42
export RESULT_ROOT="$FAIR_ROOT/results"
export POST_EVAL_ROOT="$FAIR_ROOT/post_eval"
export STUDENT_INITIALIZATION=random
export STUDENT_ADAMW_LR=1e-3
export STUDENT_ADAMW_WEIGHT_DECAY=1e-5
export STUDENT_ETA=2
export STUDENT_TEMPERATURE=20
export EVAL_WORKERS="${EVAL_WORKERS:-8}"

status="$STATUS_ROOT/${DATASET}_r41_ipc${IPC}.txt"
echo "$(date --iso-8601=seconds) fair SRe2L++ random-student IPC queue started" > "$status"

run_seed() {
    local seed="$1"
    bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" eval "$DATASET" 41 "$IPC" "$seed" \
        > "$LOG_ROOT/eval_${DATASET}_r41_ipc${IPC}_s${seed}.log" 2>&1
}

declare -a pids=()
for seed in 42 43 44; do
    run_seed "$seed" &
    pids+=("$!")
done

set +e
failures=0
for pid in "${pids[@]}"; do
    wait "$pid" || failures=$((failures + 1))
done
set -e

if (( failures != 0 )); then
    echo "$(date --iso-8601=seconds) fair SRe2L++ random-student IPC queue failed: $failures seed jobs" > "$status"
    exit 1
fi

echo "$(date --iso-8601=seconds) fair SRe2L++ random-student IPC queue complete" > "$status"
