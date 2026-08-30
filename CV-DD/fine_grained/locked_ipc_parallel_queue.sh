#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: locked_ipc_parallel_queue.sh DATASET RECOVERY_SEED IPC WAIT_FOR_RESULT}"
RECOVERY_SEED="${2:?missing recovery seed}"
IPC="${3:?missing IPC}"
WAIT_FOR_RESULT="${4:?missing prerequisite result path}"
[[ "$IPC" =~ ^(1|3|5)$ ]] || { echo "IPC must be 1, 3, or 5" >&2; exit 2; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_EXP_ROOT="${BASE_EXP_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1}"
PIPELINE_ROOT="$BASE_EXP_ROOT/diagnostics/historical_intended/pipeline"
TEACHER_DIR="$BASE_EXP_ROOT/diagnostics/historical_intended/teacher/$DATASET/tseed42"
RESULT_ROOT="$BASE_EXP_ROOT/diagnostics/student_imagenet/results"
POST_EVAL_ROOT="$BASE_EXP_ROOT/diagnostics/student_imagenet/post_eval"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
LOG_ROOT="$BASE_EXP_ROOT/diagnostics/student_imagenet/logs/jobs"
STATUS="$BASE_EXP_ROOT/diagnostics/student_imagenet/status_locked_${DATASET}_r${RECOVERY_SEED}_ipc${IPC}.txt"
mkdir -p "$LOG_ROOT"

while [[ ! -f "$WAIT_FOR_RESULT" ]]; do
    echo "$(date --iso-8601=seconds) waiting for prerequisite: $WAIT_FOR_RESULT"
    sleep "$WAIT_SECONDS"
done

export EXP_ROOT="$PIPELINE_ROOT"
export PREPARED_DATA_ROOT="$BASE_EXP_ROOT/datasets"
export TEACHER_DIR_OVERRIDE="$TEACHER_DIR"
export TEACHER_SEED=42
export RESULT_ROOT POST_EVAL_ROOT
export STUDENT_INITIALIZATION=imagenet-v1
export STUDENT_TEMPERATURE=20

echo "$(date --iso-8601=seconds) locked parallel IPC queue started" > "$STATUS"
bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" relabel \
    "$DATASET" "$RECOVERY_SEED" "$IPC" \
    > "$LOG_ROOT/relabel_${DATASET}_r${RECOVERY_SEED}_ipc${IPC}.log" 2>&1

run_eval() {
    local student_seed="$1"
    bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" eval \
        "$DATASET" "$RECOVERY_SEED" "$IPC" "$student_seed" \
        > "$LOG_ROOT/eval_${DATASET}_r${RECOVERY_SEED}_ipc${IPC}_s${student_seed}.log" 2>&1
}

# Keep two independent student evaluations resident on the selected GPU. Seed 44
# replaces seed 42 as soon as the latter finishes, while seed 43 keeps running.
run_eval 42 &
pid42=$!
run_eval 43 &
pid43=$!

set +e
wait "$pid42"
status42=$?
set -e
if (( status42 != 0 )); then
    echo "seed 42 failed with status $status42" > "$STATUS"
    wait "$pid43" || true
    exit "$status42"
fi

run_eval 44

set +e
wait "$pid43"
status43=$?
set -e
if (( status43 != 0 )); then
    echo "seed 43 failed with status $status43" > "$STATUS"
    exit "$status43"
fi

echo "$(date --iso-8601=seconds) locked parallel IPC queue complete" > "$STATUS"
