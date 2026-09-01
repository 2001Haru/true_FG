#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-launch}"
GPU_ID="${2:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STANDARD_ROOT="${STANDARD_ROOT:-/linxi/dataset/FG_SRe2L_standard/v1}"
REUSED_ROOT="${REUSED_ROOT:-$STANDARD_ROOT/sensitivity/cub_iter4000_t42_r42_s42}"
MATRIX_ROOT="${MATRIX_ROOT:-$STANDARD_ROOT/sensitivity/cub_iter4000_full_matrix_v1}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
LOG_ROOT="$MATRIX_ROOT/logs"
STATUS_ROOT="$MATRIX_ROOT/status"
LOCK_ROOT="$MATRIX_ROOT/locks"
SEEDS=(42 43 44)
IPCS=(1 3 5)
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-2}"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$LOCK_ROOT"

timestamp() { date --iso-8601=seconds; }

record_result() {
    local result="$1" teacher_seed="$2" recovery_seed="$3" ipc="$4" student_seed="$5"
    local teacher_dir="$6" recovery_root="$7" fkd_dir="$8" reused_source="${9:-}"
    local extra=()
    [[ -n "$reused_source" ]] && extra+=(--reused-source "$reused_source")
    python "$ROOT_DIR/fine_grained/record_cub4k_result.py" \
        --result "$result" --teacher-seed "$teacher_seed" \
        --recovery-seed "$recovery_seed" --ipc "$ipc" --student-seed "$student_seed" \
        --teacher-dir "$teacher_dir" --recovery-root "$recovery_root" --fkd-dir "$fkd_dir" \
        --standard-root "$STANDARD_ROOT" "${extra[@]}"
}

import_reused_arm() {
    local teacher_dir="$STANDARD_ROOT/teachers/CUB_imsize224/tseed42"
    local recovery_root="$REUSED_ROOT/pipeline/recovery/CUB_imsize224/rseed42"
    local ipc student_seed src dst fkd_dir
    for ipc in "${IPCS[@]}"; do
        fkd_dir="$REUSED_ROOT/pipeline/fkd/CUB_imsize224/rseed42/ipc${ipc}_bs20_ipc${ipc}"
        for student_seed in "${SEEDS[@]}"; do
            src="$REUSED_ROOT/results/CUB_imsize224/rseed42/ipc${ipc}_sseed${student_seed}.json"
            dst="$MATRIX_ROOT/results/tseed42/CUB_imsize224/rseed42/ipc${ipc}_sseed${student_seed}.json"
            python - "$src" "$dst" <<'PY'
import shutil
import sys
from pathlib import Path
source, target = map(Path, sys.argv[1:])
if not source.is_file():
    raise FileNotFoundError(source)
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(source, target)
PY
            record_result "$dst" 42 42 "$ipc" "$student_seed" "$teacher_dir" \
                "$recovery_root" "$fkd_dir" "$src" > /dev/null
        done
    done
    echo "$(timestamp) reused existing tseed42/rseed42 arm" \
        > "$STATUS_ROOT/CUB_imsize224_t42_r42.complete"
}

run_eval() {
    local teacher_seed="$1" recovery_seed="$2" ipc="$3" student_seed="$4"
    local task_log="$5" arm_root="$6" teacher_dir="$7"
    local result recovery_root fkd_dir log
    result="$MATRIX_ROOT/results/tseed${teacher_seed}/CUB_imsize224/rseed${recovery_seed}/ipc${ipc}_sseed${student_seed}.json"
    recovery_root="$arm_root/recovery/CUB_imsize224/rseed${recovery_seed}"
    fkd_dir="$arm_root/fkd/CUB_imsize224/rseed${recovery_seed}/ipc${ipc}_bs20_ipc${ipc}"
    log="$task_log/eval_ipc${ipc}_sseed${student_seed}.log"
    bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" eval \
        CUB_imsize224 "$recovery_seed" "$ipc" "$student_seed" > "$log" 2>&1
    record_result "$result" "$teacher_seed" "$recovery_seed" "$ipc" "$student_seed" \
        "$teacher_dir" "$recovery_root" "$fkd_dir" >> "$log" 2>&1
}

wait_batch() {
    local failed=0 pid
    for pid in "$@"; do wait "$pid" || failed=1; done
    (( failed == 0 ))
}

execute_task() {
    local teacher_seed="$1" recovery_seed="$2"
    local task_id="CUB_imsize224_t${teacher_seed}_r${recovery_seed}"
    local task_log="$LOG_ROOT/jobs/tseed${teacher_seed}/rseed${recovery_seed}"
    local arm_root="$MATRIX_ROOT/arms/tseed${teacher_seed}"
    local teacher_dir="$STANDARD_ROOT/teachers/CUB_imsize224/tseed${teacher_seed}"
    local patch_base="$STANDARD_ROOT/arms/tseed${teacher_seed}/patches/CUB_imsize224/tseed${teacher_seed}_pseed42"
    local running="$STATUS_ROOT/${task_id}.running" complete="$STATUS_ROOT/${task_id}.complete"
    local failed="$STATUS_ROOT/${task_id}.failed"
    mkdir -p "$task_log"
    echo "$(timestamp) gpu=$GPU_ID task=$task_id started" > "$running"
    rm -f "$failed"
    cleanup_task() {
        local rc=$?
        if (( rc != 0 )); then echo "$(timestamp) exit=$rc" > "$failed"; fi
        rm -f "$running"
        return "$rc"
    }
    trap cleanup_task EXIT
    [[ -f "$teacher_dir/complete.json" && -f "$teacher_dir/teacher_gate.json" ]]
    [[ -d "$patch_base/2" && -f "$patch_base/2/patch_manifest.json" ]]

    export EXP_ROOT="$arm_root"
    export PREPARED_DATA_ROOT
    export TEACHER_DIR_OVERRIDE="$teacher_dir"
    export TEACHER_SEED="$teacher_seed"
    export PATCH_SEED=42
    export PATCH_BASE_OVERRIDE="$patch_base"
    export RECOVERY_ITERATIONS_OVERRIDE=4000
    export RELABEL_WORKERS=8
    export RELABEL_MANIFEST_REQUIRED=1
    export EVAL_WORKERS=8
    export EVAL_PERSISTENT_WORKERS=1
    export STUDENT_INITIALIZATION=random
    export STUDENT_ADAMW_LR=1e-3
    export STUDENT_ADAMW_WEIGHT_DECAY=1e-5
    export STUDENT_ETA=2
    export STUDENT_TEMPERATURE=20
    export RESULT_ROOT="$MATRIX_ROOT/results/tseed${teacher_seed}"
    export POST_EVAL_ROOT="$MATRIX_ROOT/post_eval/tseed${teacher_seed}"

    bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" recover \
        CUB_imsize224 "$recovery_seed" > "$task_log/recover.log" 2>&1
    bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" sample \
        CUB_imsize224 "$recovery_seed" > "$task_log/sample.log" 2>&1
    local ipc student_seed pid pids=()
    for ipc in "${IPCS[@]}"; do
        bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" relabel \
            CUB_imsize224 "$recovery_seed" "$ipc" > "$task_log/relabel_ipc${ipc}.log" 2>&1
    done
    for ipc in "${IPCS[@]}"; do
        for student_seed in "${SEEDS[@]}"; do
            run_eval "$teacher_seed" "$recovery_seed" "$ipc" "$student_seed" \
                "$task_log" "$arm_root" "$teacher_dir" &
            pids+=("$!")
            if (( ${#pids[@]} == EVAL_CONCURRENCY )); then
                wait_batch "${pids[@]}"
                pids=()
            fi
        done
    done
    if (( ${#pids[@]} > 0 )); then wait_batch "${pids[@]}"; fi
    echo "$(timestamp) gpu=$GPU_ID task=$task_id complete" > "$complete"
    rm -f "$running"
    trap - EXIT
}

try_task() {
    local teacher_seed="$1" recovery_seed="$2"
    local task_id="CUB_imsize224_t${teacher_seed}_r${recovery_seed}"
    (
        flock -n 7 || exit 75
        [[ -f "$STATUS_ROOT/${task_id}.complete" ]] && exit 0
        execute_task "$teacher_seed" "$recovery_seed"
    ) 7>"$LOCK_ROOT/${task_id}.lock"
}

worker_main() {
    [[ "$GPU_ID" =~ ^[01]$ ]] || { echo "worker requires GPU 0 or 1" >&2; exit 2; }
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    export TORCH_HOME=/linxi/dataset/FD2/torch_cache
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
    echo "$(timestamp) worker gpu=$GPU_ID started"
    while true; do
        local remaining=0 claimed=0 teacher_seed recovery_seed rc
        for recovery_seed in "${SEEDS[@]}"; do
            for teacher_seed in "${SEEDS[@]}"; do
                [[ -f "$STATUS_ROOT/CUB_imsize224_t${teacher_seed}_r${recovery_seed}.complete" ]] && continue
                remaining=$((remaining + 1))
                set +e
                try_task "$teacher_seed" "$recovery_seed"
                rc=$?
                set -e
                if (( rc == 0 )); then claimed=1; break 2; fi
                if (( rc != 75 )); then return "$rc"; fi
            done
        done
        if (( remaining == 0 )); then echo "$(timestamp) worker gpu=$GPU_ID complete"; return 0; fi
        if (( claimed == 0 )); then sleep 30; fi
    done
}

launcher_main() {
    exec 9>"$LOCK_ROOT/launcher.lock"
    flock -n 9 || { echo "CUB 4k full matrix already running" >&2; exit 1; }
    python - "$MATRIX_ROOT" "$STANDARD_ROOT" "$REUSED_ROOT" <<'PY'
import json
import os
import sys
from pathlib import Path
root, standard, reused = map(Path, sys.argv[1:])
payload = {
    'status': 'running', 'experiment': 'cub_recovery_4000_full_crossed_seed_matrix',
    'teacher_seeds': [42,43,44], 'recovery_seeds': [42,43,44],
    'student_seeds': [42,43,44], 'ipcs': [1,3,5],
    'recovery_iterations': 4000, 'expected_recovery_arms': 9,
    'expected_results': 81, 'reused_arm': 'tseed42/rseed42',
    'reused_root': str(reused.resolve()), 'paired_standard_root': str(standard.resolve()),
}
path = root / 'matrix_definition.json'; path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix('.json.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
os.replace(tmp,path)
PY
    import_reused_arm
    python "$ROOT_DIR/fine_grained/summarize_cub4k_full_matrix.py" \
        --matrix-root "$MATRIX_ROOT" --standard-root "$STANDARD_ROOT" \
        > "$LOG_ROOT/summary_initial.log" 2>&1
    echo "$(timestamp) launcher started" > "$STATUS_ROOT/launcher.running"
    bash "$0" worker 0 > "$LOG_ROOT/worker_gpu0.log" 2>&1 & local p0=$!
    bash "$0" worker 1 > "$LOG_ROOT/worker_gpu1.log" 2>&1 & local p1=$!
    local failed=0
    wait "$p0" || failed=1
    wait "$p1" || failed=1
    python "$ROOT_DIR/fine_grained/summarize_cub4k_full_matrix.py" \
        --matrix-root "$MATRIX_ROOT" --standard-root "$STANDARD_ROOT" \
        > "$LOG_ROOT/summary_final.log" 2>&1
    rm -f "$STATUS_ROOT/launcher.running"
    if (( failed )); then echo "$(timestamp) launcher failed" > "$STATUS_ROOT/launcher.failed"; return 1; fi
    echo "$(timestamp) launcher complete" > "$STATUS_ROOT/launcher.complete"
}

case "$MODE" in
    launch) launcher_main ;;
    worker) worker_main ;;
    *) echo "usage: $0 [launch|worker GPU]" >&2; exit 2 ;;
esac
