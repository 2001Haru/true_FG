#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STANDARD_ROOT="${STANDARD_ROOT:-/linxi/dataset/FG_SRe2L_standard/v1}"
SENSITIVITY_ROOT="${SENSITIVITY_ROOT:-$STANDARD_ROOT/sensitivity/cub_iter4000_t42_r42_s42}"
PIPELINE_ROOT="$SENSITIVITY_ROOT/pipeline"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
TEACHER_DIR="$STANDARD_ROOT/teachers/CUB_imsize224/tseed42"
LOG_ROOT="$SENSITIVITY_ROOT/logs"
STATUS_ROOT="$SENSITIVITY_ROOT/status"
RESULT_ROOT="$SENSITIVITY_ROOT/results"
POST_EVAL_ROOT="$SENSITIVITY_ROOT/post_eval"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT"

cleanup() {
    local rc=$?
    if (( rc != 0 )); then
        echo "$(date --iso-8601=seconds) extra Student seeds failed exit=$rc" \
            > "$STATUS_ROOT/extra_students.failed"
    fi
    rm -f "$STATUS_ROOT/extra_students.running"
    return "$rc"
}
trap cleanup EXIT

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export TORCH_HOME="${TORCH_HOME:-/linxi/dataset/FD2/torch_cache}"
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"
export EXP_ROOT="$PIPELINE_ROOT"
export PREPARED_DATA_ROOT
export TEACHER_DIR_OVERRIDE="$TEACHER_DIR"
export TEACHER_SEED=42
export PATCH_SEED=42
export RELABEL_WORKERS=8
export RELABEL_MANIFEST_REQUIRED=1
export EVAL_WORKERS=8
export EVAL_PERSISTENT_WORKERS=1
export STUDENT_INITIALIZATION=random
export STUDENT_ADAMW_LR=1e-3
export STUDENT_ADAMW_WEIGHT_DECAY=1e-5
export STUDENT_ETA=2
export STUDENT_TEMPERATURE=20
export RESULT_ROOT POST_EVAL_ROOT

exec 9>"$STATUS_ROOT/extra_students.lock"
flock -n 9 || { echo "extra Student-seed queue already running" >&2; exit 1; }
for ipc in 1 3 5; do
    [[ -f "$PIPELINE_ROOT/fkd/CUB_imsize224/rseed42/ipc${ipc}_bs20_ipc${ipc}/fkd_audit.json" ]] || {
        echo "missing completed IPC${ipc} FKD" >&2; exit 1;
    }
done
echo "$(date --iso-8601=seconds) extra Student seeds started" > "$STATUS_ROOT/extra_students.running"

run_eval() {
    local gpu="$1" ipc="$2" student_seed="$3"
    CUDA_VISIBLE_DEVICES="$gpu" bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" eval \
        CUB_imsize224 42 "$ipc" "$student_seed" \
        > "$LOG_ROOT/eval_ipc${ipc}_sseed${student_seed}.log" 2>&1
}

run_batch() {
    local failed=0 pid
    for pid in "$@"; do wait "$pid" || failed=1; done
    (( failed == 0 ))
}

run_eval 0 1 43 & p1=$!
run_eval 0 1 44 & p2=$!
run_eval 1 3 43 & p3=$!
run_eval 1 3 44 & p4=$!
run_batch "$p1" "$p2" "$p3" "$p4"
run_eval 0 5 43 & p5=$!
run_eval 1 5 44 & p6=$!
run_batch "$p5" "$p6"

python "$ROOT_DIR/fine_grained/summarize_cub_iteration_sensitivity.py" \
    --sensitivity-root "$SENSITIVITY_ROOT" --standard-root "$STANDARD_ROOT" \
    > "$LOG_ROOT/summary_3_student_seeds.log" 2>&1
rm -f "$STATUS_ROOT/extra_students.running"
echo "$(date --iso-8601=seconds) extra Student seeds complete" \
    > "$STATUS_ROOT/extra_students.complete"
trap - EXIT
