#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STANDARD_ROOT="${STANDARD_ROOT:-/linxi/dataset/FG_SRe2L_standard/v1}"
FOUR_K_ROOT="${FOUR_K_ROOT:-$STANDARD_ROOT/sensitivity/cub_iter4000_t42_r42_s42}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$STANDARD_ROOT/sensitivity/cub_imagenet_student_iter4k_vs10k_t42_r42}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
TEACHER_DIR="$STANDARD_ROOT/teachers/CUB_imsize224/tseed42"
LOG_ROOT="$EXPERIMENT_ROOT/logs"
STATUS_ROOT="$EXPERIMENT_ROOT/status"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT"

cleanup() {
    local rc=$?
    if (( rc != 0 )); then
        echo "$(date --iso-8601=seconds) ImageNet Student matrix failed exit=$rc" \
            > "$STATUS_ROOT/launcher.failed"
    fi
    rm -f "$STATUS_ROOT/launcher.running"
    return "$rc"
}
trap cleanup EXIT

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export TORCH_HOME="${TORCH_HOME:-/linxi/dataset/FD2/torch_cache}"
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"
exec 9>"$STATUS_ROOT/launcher.lock"
flock -n 9 || { echo "ImageNet Student matrix already running" >&2; exit 1; }

for ipc in 1 3 5; do
    [[ -f "$STANDARD_ROOT/arms/tseed42/fkd/CUB_imsize224/rseed42/ipc${ipc}_bs20_ipc${ipc}/fkd_audit.json" ]]
    [[ -f "$FOUR_K_ROOT/pipeline/fkd/CUB_imsize224/rseed42/ipc${ipc}_bs20_ipc${ipc}/fkd_audit.json" ]]
done
echo "$(date --iso-8601=seconds) ImageNet Student matrix started" > "$STATUS_ROOT/launcher.running"

variant_worker() {
    local gpu="$1" iterations="$2" pipeline result_root post_root
    if [[ "$iterations" == 4000 ]]; then
        pipeline="$FOUR_K_ROOT/pipeline"
    else
        pipeline="$STANDARD_ROOT/arms/tseed42"
    fi
    result_root="$EXPERIMENT_ROOT/results/iter${iterations}"
    post_root="$EXPERIMENT_ROOT/post_eval/iter${iterations}"
    run_one() {
        local ipc="$1" student_seed="$2"
        CUDA_VISIBLE_DEVICES="$gpu" EXP_ROOT="$pipeline" \
            PREPARED_DATA_ROOT="$PREPARED_DATA_ROOT" \
            TEACHER_DIR_OVERRIDE="$TEACHER_DIR" TEACHER_SEED=42 PATCH_SEED=42 \
            RESULT_ROOT="$result_root" POST_EVAL_ROOT="$post_root" \
            EVAL_WORKERS=8 EVAL_PERSISTENT_WORKERS=1 \
            STUDENT_INITIALIZATION=imagenet-v1 STUDENT_ADAMW_LR=1e-3 \
            STUDENT_ADAMW_WEIGHT_DECAY=1e-5 STUDENT_ETA=2 STUDENT_TEMPERATURE=20 \
            bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" eval \
            CUB_imsize224 42 "$ipc" "$student_seed" \
            > "$LOG_ROOT/iter${iterations}_ipc${ipc}_sseed${student_seed}.log" 2>&1
    }
    local pids=() ipc student_seed pid failed
    for ipc in 1 3 5; do
        for student_seed in 42 43 44; do
            run_one "$ipc" "$student_seed" &
            pids+=("$!")
            if (( ${#pids[@]} == 2 )); then
                failed=0
                for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
                (( failed == 0 )) || return 1
                pids=()
            fi
        done
    done
    if (( ${#pids[@]} > 0 )); then
        failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
        (( failed == 0 )) || return 1
    fi
}

variant_worker 0 10000 & pid10=$!
variant_worker 1 4000 & pid4=$!
failed=0
wait "$pid10" || failed=1
wait "$pid4" || failed=1
(( failed == 0 )) || exit 1

python "$ROOT_DIR/fine_grained/summarize_cub_imagenet_student_matrix.py" \
    --experiment-root "$EXPERIMENT_ROOT" --standard-root "$STANDARD_ROOT" \
    --four-k-root "$FOUR_K_ROOT" > "$LOG_ROOT/summary.log" 2>&1
rm -f "$STATUS_ROOT/launcher.running"
echo "$(date --iso-8601=seconds) ImageNet Student matrix complete" > "$STATUS_ROOT/launcher.complete"
trap - EXIT
