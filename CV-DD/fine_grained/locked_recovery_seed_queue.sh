#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: locked_recovery_seed_queue.sh DATASET RECOVERY_SEED WAIT_FOR_RESULT}"
RECOVERY_SEED="${2:?missing recovery seed}"
WAIT_FOR_RESULT="${3:?missing prerequisite result path}"
[[ "$RECOVERY_SEED" =~ ^[0-9]+$ ]] || { echo "recovery seed must be numeric" >&2; exit 2; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_EXP_ROOT="${BASE_EXP_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1}"
PIPELINE_ROOT="$BASE_EXP_ROOT/diagnostics/historical_intended/pipeline"
TEACHER_DIR="$BASE_EXP_ROOT/diagnostics/historical_intended/teacher/$DATASET/tseed42"
RESULT_ROOT="$BASE_EXP_ROOT/diagnostics/student_imagenet/results"
POST_EVAL_ROOT="$BASE_EXP_ROOT/diagnostics/student_imagenet/post_eval"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
LOG_ROOT="$BASE_EXP_ROOT/diagnostics/student_imagenet/logs/jobs"
STATUS="$BASE_EXP_ROOT/diagnostics/student_imagenet/status_locked_${DATASET}_r${RECOVERY_SEED}.txt"
mkdir -p "$LOG_ROOT"

while [[ ! -f "$WAIT_FOR_RESULT" ]]; do
    echo "$(date --iso-8601=seconds) waiting for prerequisite: $WAIT_FOR_RESULT"
    sleep "$WAIT_SECONDS"
done

export EXP_ROOT="$PIPELINE_ROOT"
export PREPARED_DATA_ROOT="$BASE_EXP_ROOT/datasets"
export TEACHER_DIR_OVERRIDE="$TEACHER_DIR"
export TEACHER_SEED=42
export PATCH_SEED=42
export RESULT_ROOT POST_EVAL_ROOT
export STUDENT_INITIALIZATION=imagenet-v1
export STUDENT_TEMPERATURE=20

echo "$(date --iso-8601=seconds) recovery-seed queue started" > "$STATUS"
bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" recover \
    "$DATASET" "$RECOVERY_SEED" \
    > "$LOG_ROOT/recover_${DATASET}_r${RECOVERY_SEED}.log" 2>&1
bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" sample \
    "$DATASET" "$RECOVERY_SEED" \
    > "$LOG_ROOT/sample_${DATASET}_r${RECOVERY_SEED}.log" 2>&1

# Student seed 42 is fixed here. Combined with the rseed41 × sseed42/43/44
# matrix, this isolates recovery variation without an unnecessary full 3×3
# recovery/student seed factorial.
for ipc in 1 3 5; do
    bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" relabel \
        "$DATASET" "$RECOVERY_SEED" "$ipc" \
        > "$LOG_ROOT/relabel_${DATASET}_r${RECOVERY_SEED}_ipc${ipc}.log" 2>&1
    bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" eval \
        "$DATASET" "$RECOVERY_SEED" "$ipc" 42 \
        > "$LOG_ROOT/eval_${DATASET}_r${RECOVERY_SEED}_ipc${ipc}_s42.log" 2>&1
done

echo "$(date --iso-8601=seconds) recovery-seed queue complete" > "$STATUS"
