#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-launch}"
GPU_ID="${2:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STANDARD_ROOT="${STANDARD_ROOT:-/linxi/dataset/FG_SRe2L_standard/v1}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
LOG_ROOT="$STANDARD_ROOT/logs"
JOB_LOG_ROOT="$LOG_ROOT/jobs"
STATUS_ROOT="$STANDARD_ROOT/status"
LOCK_ROOT="$STANDARD_ROOT/locks"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-2}"
DATASETS=(CUB_imsize224 A_imsize224 SC_imsize224)
SEEDS=(42 43 44)
IPCS=(1 3 5)
mkdir -p "$LOG_ROOT" "$JOB_LOG_ROOT" "$STATUS_ROOT" "$LOCK_ROOT"

timestamp() { date --iso-8601=seconds; }

task_exit_trap() {
    local rc=$?
    if (( rc != 0 )); then
        echo "$(timestamp) gpu=$GPU_ID task=$TASK_ID exit=$rc" > "$TASK_FAILED"
    fi
    rm -f "$TASK_RUNNING"
    return "$rc"
}

write_definition() {
    local revision
    revision="$(git -C "$ROOT_DIR" rev-parse HEAD)"
    python - "$STANDARD_ROOT" "$revision" <<'PY'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
revision = sys.argv[2]
datasets = ["CUB_imsize224", "A_imsize224", "SC_imsize224"]
seeds = [42, 43, 44]
ipcs = [1, 3, 5]
expected = []
for dataset in datasets:
    for teacher_seed in seeds:
        for recovery_seed in seeds:
            for ipc in ipcs:
                for student_seed in seeds:
                    expected.append(str((
                        root / "results" / f"tseed{teacher_seed}" / dataset /
                        f"rseed{recovery_seed}" / f"ipc{ipc}_sseed{student_seed}.json"
                    ).resolve()))
payload = {
    "status": "running",
    "protocol_name": "standard_protocol",
    "protocol_version": "v1",
    "git_revision": revision,
    "datasets": datasets,
    "teacher_seeds": seeds,
    "recovery_seeds": seeds,
    "student_seeds": seeds,
    "ipcs": ipcs,
    "expected_results": len(expected),
    "legacy_results_status": "exploratory_only",
    "standard_root": str(root.resolve()),
    "log_root": str((root / "logs").resolve()),
    "result_root": str((root / "results").resolve()),
    "summary": str((root / "summary" / "standard_matrix.json").resolve()),
    "expected_result_files": expected,
}
path = root / "matrix_definition.json"
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

ensure_teacher() {
    local dataset="$1" teacher_seed="$2" task_log="$3"
    local teacher_dir="$STANDARD_ROOT/teachers/$dataset/tseed${teacher_seed}"
    local lock="$LOCK_ROOT/teacher_${dataset}_t${teacher_seed}.lock"
    (
        flock 8
        if [[ ! -f "$teacher_dir/complete.json" ]]; then
            echo "$(timestamp) teacher start dataset=$dataset tseed=$teacher_seed"
            EXP_ROOT="$STANDARD_ROOT" PREPARED_DATA_ROOT="$PREPARED_DATA_ROOT" \
                TEACHER_WORKERS=8 \
                bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" teacher \
                "$dataset" "$teacher_seed" \
                > "$task_log/teacher_t${teacher_seed}.log" 2>&1
        fi
        if ! python "$ROOT_DIR/fine_grained/check_teacher_gate.py" \
            --dataset-name "$dataset" --teacher-dir "$teacher_dir" \
            > "$task_log/teacher_gate_t${teacher_seed}.log" 2>&1; then
            echo "$(timestamp) WARNING teacher gate below CAL-reference threshold; standard crossed-seed arm continues"
        fi
        [[ -f "$teacher_dir/teacher_gate.json" ]]
    ) 8>"$lock"
}

ensure_patches() {
    local dataset="$1" teacher_seed="$2" task_log="$3" arm_root="$4"
    local teacher_dir="$STANDARD_ROOT/teachers/$dataset/tseed${teacher_seed}"
    local lock="$LOCK_ROOT/patch_${dataset}_t${teacher_seed}.lock"
    (
        flock 8
        EXP_ROOT="$arm_root" PREPARED_DATA_ROOT="$PREPARED_DATA_ROOT" \
            TEACHER_DIR_OVERRIDE="$teacher_dir" TEACHER_SEED="$teacher_seed" PATCH_SEED=42 \
            bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" patches "$dataset" 42 \
            > "$task_log/patches_t${teacher_seed}.log" 2>&1
    ) 8>"$lock"
}

run_eval() {
    local dataset="$1" teacher_seed="$2" recovery_seed="$3" ipc="$4" student_seed="$5"
    local task_log="$6" arm_root="$7" teacher_dir="$8"
    local batch fkd_dir recovery_root result log
    if [[ "$dataset" == SC_imsize224 ]]; then batch=14; else batch=20; fi
    fkd_dir="$arm_root/fkd/$dataset/rseed${recovery_seed}/ipc${ipc}_bs${batch}_ipc${ipc}"
    recovery_root="$arm_root/recovery/$dataset/rseed${recovery_seed}"
    result="$STANDARD_ROOT/results/tseed${teacher_seed}/$dataset/rseed${recovery_seed}/ipc${ipc}_sseed${student_seed}.json"
    log="$task_log/eval_ipc${ipc}_sseed${student_seed}.log"
    bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" eval \
        "$dataset" "$recovery_seed" "$ipc" "$student_seed" > "$log" 2>&1
    python "$ROOT_DIR/fine_grained/record_standard_result.py" \
        --result "$result" --dataset "$dataset" \
        --teacher-seed "$teacher_seed" --recovery-seed "$recovery_seed" \
        --ipc "$ipc" --student-seed "$student_seed" \
        --teacher-dir "$teacher_dir" --recovery-root "$recovery_root" \
        --fkd-dir "$fkd_dir" >> "$log" 2>&1
    if [[ "$dataset" == CUB_imsize224 ]]; then
        python "$ROOT_DIR/fine_grained/audit_result.py" --result "$result" \
            --classes 200 --validation-images 5794 >> "$log" 2>&1
    elif [[ "$dataset" == A_imsize224 ]]; then
        python "$ROOT_DIR/fine_grained/audit_result.py" --result "$result" \
            --classes 100 --validation-images 3333 >> "$log" 2>&1
    else
        python "$ROOT_DIR/fine_grained/audit_result.py" --result "$result" \
            --classes 196 --validation-images 8041 >> "$log" 2>&1
    fi
}

run_eval_batch() {
    local failed=0 pid
    for pid in "$@"; do
        if ! wait "$pid"; then failed=1; fi
    done
    (( failed == 0 ))
}

execute_task() {
    local dataset="$1" teacher_seed="$2" recovery_seed="$3"
    local task_id="${dataset}_t${teacher_seed}_r${recovery_seed}"
    local task_log="$JOB_LOG_ROOT/$dataset/tseed${teacher_seed}/rseed${recovery_seed}"
    local arm_root="$STANDARD_ROOT/arms/tseed${teacher_seed}"
    local teacher_dir="$STANDARD_ROOT/teachers/$dataset/tseed${teacher_seed}"
    local running="$STATUS_ROOT/${task_id}.running"
    local complete="$STATUS_ROOT/${task_id}.complete"
    local failed="$STATUS_ROOT/${task_id}.failed"
    mkdir -p "$task_log"
    echo "$(timestamp) gpu=$GPU_ID task=$task_id started" > "$running"
    rm -f "$failed"
    TASK_ID="$task_id"
    TASK_RUNNING="$running"
    TASK_FAILED="$failed"
    trap task_exit_trap EXIT

    ensure_teacher "$dataset" "$teacher_seed" "$task_log"
    ensure_patches "$dataset" "$teacher_seed" "$task_log" "$arm_root"

    export EXP_ROOT="$arm_root"
    export PREPARED_DATA_ROOT
    export TEACHER_DIR_OVERRIDE="$teacher_dir"
    export TEACHER_SEED="$teacher_seed"
    export PATCH_SEED=42
    export TEACHER_WORKERS=8
    export RELABEL_WORKERS=8
    export RELABEL_MANIFEST_REQUIRED=1
    export EVAL_WORKERS=8
    export EVAL_PERSISTENT_WORKERS=1
    export STUDENT_INITIALIZATION=random
    export STUDENT_ADAMW_LR=1e-3
    export STUDENT_ADAMW_WEIGHT_DECAY=1e-5
    export STUDENT_ETA=2
    export STUDENT_TEMPERATURE=20
    export RESULT_ROOT="$STANDARD_ROOT/results/tseed${teacher_seed}"
    export POST_EVAL_ROOT="$STANDARD_ROOT/post_eval/tseed${teacher_seed}"

    bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" recover \
        "$dataset" "$recovery_seed" > "$task_log/recover.log" 2>&1
    bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" sample \
        "$dataset" "$recovery_seed" > "$task_log/sample.log" 2>&1
    for ipc in "${IPCS[@]}"; do
        bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" relabel \
            "$dataset" "$recovery_seed" "$ipc" > "$task_log/relabel_ipc${ipc}.log" 2>&1
    done

    local pids=()
    for ipc in "${IPCS[@]}"; do
        for student_seed in "${SEEDS[@]}"; do
            run_eval "$dataset" "$teacher_seed" "$recovery_seed" "$ipc" \
                "$student_seed" "$task_log" "$arm_root" "$teacher_dir" &
            pids+=("$!")
            if (( ${#pids[@]} == EVAL_CONCURRENCY )); then
                run_eval_batch "${pids[@]}"
                pids=()
            fi
        done
    done
    if (( ${#pids[@]} > 0 )); then run_eval_batch "${pids[@]}"; fi

    echo "$(timestamp) gpu=$GPU_ID task=$task_id complete" > "$complete"
    rm -f "$running"
    trap - EXIT
}

try_task() {
    local dataset="$1" teacher_seed="$2" recovery_seed="$3"
    local task_id="${dataset}_t${teacher_seed}_r${recovery_seed}"
    local lock="$LOCK_ROOT/task_${task_id}.lock"
    (
        flock -n 7 || exit 75
        [[ -f "$STATUS_ROOT/${task_id}.complete" ]] && exit 0
        execute_task "$dataset" "$teacher_seed" "$recovery_seed"
    ) 7>"$lock"
}

worker_main() {
    [[ "$GPU_ID" =~ ^[01]$ ]] || { echo "worker requires GPU 0 or 1" >&2; exit 2; }
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    export TORCH_HOME=/linxi/dataset/FD2/torch_cache
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
    echo "$(timestamp) worker gpu=$GPU_ID started"
    while true; do
        local remaining=0 claimed=0 dataset teacher_seed recovery_seed rc
        for recovery_seed in "${SEEDS[@]}"; do
            for teacher_seed in "${SEEDS[@]}"; do
                for dataset in "${DATASETS[@]}"; do
                    [[ -f "$STATUS_ROOT/${dataset}_t${teacher_seed}_r${recovery_seed}.complete" ]] && continue
                    remaining=$((remaining + 1))
                    set +e
                    try_task "$dataset" "$teacher_seed" "$recovery_seed"
                    rc=$?
                    set -e
                    if (( rc == 0 )); then claimed=1; break 3; fi
                    if (( rc != 75 )); then return "$rc"; fi
                done
            done
        done
        if (( remaining == 0 )); then
            echo "$(timestamp) worker gpu=$GPU_ID complete"
            return 0
        fi
        if (( claimed == 0 )); then sleep 30; fi
    done
}

launcher_main() {
    command -v flock >/dev/null 2>&1 || { echo "flock is required" >&2; exit 1; }
    exec 9>"$LOCK_ROOT/launcher.lock"
    flock -n 9 || { echo "standard matrix launcher is already running" >&2; exit 1; }
    write_definition
    for dataset in "${DATASETS[@]}"; do
        [[ -d "$PREPARED_DATA_ROOT/$dataset/train" && -d "$PREPARED_DATA_ROOT/$dataset/test" ]] || {
            echo "missing prepared dataset: $PREPARED_DATA_ROOT/$dataset" >&2; exit 1;
        }
        EXP_ROOT="$STANDARD_ROOT" PREPARED_DATA_ROOT="$PREPARED_DATA_ROOT" \
            bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" audit "$dataset" 42 \
            > "$LOG_ROOT/audit_${dataset}.log" 2>&1
    done
    python "$ROOT_DIR/fine_grained/summarize_standard_matrix.py" \
        --standard-root "$STANDARD_ROOT" > "$LOG_ROOT/summary_initial.log" 2>&1
    echo "$(timestamp) launcher started" > "$STATUS_ROOT/launcher.running"
    bash "$0" worker 0 > "$LOG_ROOT/worker_gpu0.log" 2>&1 &
    local pid0=$!
    bash "$0" worker 1 > "$LOG_ROOT/worker_gpu1.log" 2>&1 &
    local pid1=$!
    local failed=0
    wait "$pid0" || failed=1
    wait "$pid1" || failed=1
    python "$ROOT_DIR/fine_grained/summarize_standard_matrix.py" \
        --standard-root "$STANDARD_ROOT" > "$LOG_ROOT/summary_final.log" 2>&1
    rm -f "$STATUS_ROOT/launcher.running"
    if (( failed )); then
        echo "$(timestamp) launcher failed" > "$STATUS_ROOT/launcher.failed"
        return 1
    fi
    echo "$(timestamp) launcher complete" > "$STATUS_ROOT/launcher.complete"
}

case "$MODE" in
    launch) launcher_main ;;
    worker) worker_main ;;
    *) echo "usage: $0 [launch | worker GPU_ID]" >&2; exit 2 ;;
esac
