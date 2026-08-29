#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: recovery_queue.sh DATASET BUILD_PATCHES RECOVERY_SEED...}"
BUILD_PATCHES="${2:?missing BUILD_PATCHES (0 or 1)}"
shift 2
[[ "$#" -gt 0 ]] || { echo "At least one recovery seed is required" >&2; exit 2; }
[[ "$BUILD_PATCHES" =~ ^(0|1)$ ]] || { echo "BUILD_PATCHES must be 0 or 1" >&2; exit 2; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
TEACHER_SEED="${TEACHER_SEED:-42}"
PATCH_SEED="${PATCH_SEED:-42}"
LOG_ROOT="$EXP_ROOT/logs/jobs"
STATUS_ROOT="$EXP_ROOT/queue_status"
TEACHER_COMPLETE="$EXP_ROOT/teachers/$DATASET/tseed${TEACHER_SEED}/complete.json"
TEACHER_DIR="$EXP_ROOT/teachers/$DATASET/tseed${TEACHER_SEED}"
PATCH_COMPLETE="$EXP_ROOT/patches/$DATASET/tseed${TEACHER_SEED}_pseed${PATCH_SEED}/2/patch_manifest.json"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT"

wait_for_file() {
    local path="$1" description="$2"
    while [[ ! -f "$path" ]]; do
        echo "$(date --iso-8601=seconds) waiting for $description: $path" >&2
        sleep "$WAIT_SECONDS"
    done
}

wait_for_file "$TEACHER_COMPLETE" "teacher completion"
if [[ -n "${ADDITIONAL_WAIT_FILE:-}" ]]; then
    wait_for_file "$ADDITIONAL_WAIT_FILE" "additional GPU-owner completion"
fi
python "$ROOT_DIR/fine_grained/check_teacher_gate.py" \
    --dataset-name "$DATASET" --teacher-dir "$TEACHER_DIR" \
    --tolerance "${TEACHER_GATE_TOLERANCE:-3.0}"

if [[ "$BUILD_PATCHES" == 1 && ! -f "$PATCH_COMPLETE" ]]; then
    echo "$(date --iso-8601=seconds) generating patches for $DATASET"
    bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" patches "$DATASET" "$PATCH_SEED" \
        > "$LOG_ROOT/patches_${DATASET}_tseed${TEACHER_SEED}_pseed${PATCH_SEED}.log" 2>&1
fi
wait_for_file "$PATCH_COMPLETE" "patch completion"

for recovery_seed in "$@"; do
    status="$STATUS_ROOT/${DATASET}_rseed${recovery_seed}.status"
    echo "$(date --iso-8601=seconds) recovery started" > "$status"
    echo "$(date --iso-8601=seconds) recovering $DATASET seed $recovery_seed"
    bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" recover "$DATASET" "$recovery_seed" \
        > "$LOG_ROOT/recover_${DATASET}_rseed${recovery_seed}.log" 2>&1
    bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" sample "$DATASET" "$recovery_seed" \
        > "$LOG_ROOT/sample_${DATASET}_rseed${recovery_seed}.log" 2>&1
    echo "$(date --iso-8601=seconds) recovery and IPC1/3 sampling complete" > "$status"
    echo "$(date --iso-8601=seconds) completed $DATASET seed $recovery_seed"
done

echo "$(date --iso-8601=seconds) queue complete: $DATASET seeds $*"
