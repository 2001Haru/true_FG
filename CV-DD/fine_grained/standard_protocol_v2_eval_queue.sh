#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-launch}"
GPU_ID="${2:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V1_ROOT="${V1_ROOT:-/linxi/dataset/FG_SRe2L_standard/v1}"
CUB4K_ROOT="${CUB4K_ROOT:-$V1_ROOT/sensitivity/cub_iter4000_full_matrix_v1}"
CUB_REUSED_ROOT="${CUB_REUSED_ROOT:-$V1_ROOT/sensitivity/cub_iter4000_t42_r42_s42}"
V2_ROOT="${V2_ROOT:-/linxi/dataset/FG_SRe2L_standard/v2}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-2}"
GPUS="${GPUS:-0 1}"
SEEDS=(42 43 44)
IPCS=(1 3 5)
DATASETS=(CUB_imsize224 A_imsize224 SC_imsize224)
LOG_ROOT="$V2_ROOT/logs"
STATUS_ROOT="$V2_ROOT/status"
LOCK_ROOT="$V2_ROOT/locks"
mkdir -p "$LOG_ROOT/jobs" "$STATUS_ROOT" "$LOCK_ROOT"

timestamp() { date --iso-8601=seconds; }

upstream_root() {
    local dataset="$1" teacher_seed="$2" recovery_seed="$3"
    if [[ "$dataset" != CUB_imsize224 ]]; then
        printf '%s\n' "$V1_ROOT/arms/tseed${teacher_seed}"
    elif [[ "$teacher_seed" == 42 && "$recovery_seed" == 42 ]]; then
        printf '%s\n' "$CUB_REUSED_ROOT/pipeline"
    else
        printf '%s\n' "$CUB4K_ROOT/arms/tseed${teacher_seed}"
    fi
}

reference_result() {
    local dataset="$1" teacher_seed="$2" recovery_seed="$3" ipc="$4" student_seed="$5"
    if [[ "$dataset" == CUB_imsize224 ]]; then
        printf '%s\n' "$CUB4K_ROOT/results/tseed${teacher_seed}/$dataset/rseed${recovery_seed}/ipc${ipc}_sseed${student_seed}.json"
    else
        printf '%s\n' "$V1_ROOT/results/tseed${teacher_seed}/$dataset/rseed${recovery_seed}/ipc${ipc}_sseed${student_seed}.json"
    fi
}

write_definition() {
    local revision
    revision="$(git -C "$ROOT_DIR" rev-parse HEAD)"
    python - "$V2_ROOT" "$V1_ROOT" "$CUB4K_ROOT" "$revision" <<'PY'
import json, os, sys
from pathlib import Path
root, v1, cub4k = map(Path, sys.argv[1:4]); revision = sys.argv[4]
datasets = ["CUB_imsize224", "A_imsize224", "SC_imsize224"]
seeds = [42, 43, 44]; ipcs = [1, 3, 5]
expected = [str((root / "results" / f"tseed{t}" / d / f"rseed{r}" / f"ipc{i}_sseed{s}.json").resolve())
            for d in datasets for t in seeds for r in seeds for i in ipcs for s in seeds]
payload = {
    "status": "prepared_not_started", "protocol_name": "fine_grained_sre2l_standard_protocol",
    "protocol_version": "v2", "git_revision": revision, "datasets": datasets,
    "teacher_seeds": seeds, "recovery_seeds": seeds, "student_seeds": seeds,
    "ipcs": ipcs, "recovery_iterations": {d: 4000 for d in datasets},
    "expected_results": len(expected), "new_student_trainings": len(expected),
    "upstream_policy": "reuse audited v1 Teacher/Recovery/FKD; CUB uses full 4k matrix",
    "v1_root": str(v1.resolve()), "cub4k_root": str(cub4k.resolve()),
    "result_root": str((root / "results").resolve()),
    "summary": str((root / "summary/standard_v2_matrix.json").resolve()),
    "expected_result_files": expected,
}
path = root / "matrix_definition.json"; path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(".json.tmp"); tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.replace(tmp, path)
PY
}

run_eval() {
    local dataset="$1" teacher_seed="$2" recovery_seed="$3" ipc="$4" student_seed="$5" task_log="$6"
    local upstream reference result log
    upstream="$(upstream_root "$dataset" "$teacher_seed" "$recovery_seed")"
    reference="$(reference_result "$dataset" "$teacher_seed" "$recovery_seed" "$ipc" "$student_seed")"
    result="$V2_ROOT/results/tseed${teacher_seed}/$dataset/rseed${recovery_seed}/ipc${ipc}_sseed${student_seed}.json"
    log="$task_log/eval_ipc${ipc}_sseed${student_seed}.log"
    [[ -f "$reference" ]] || { echo "missing reference: $reference" >&2; return 1; }
    EXP_ROOT="$upstream" PREPARED_DATA_ROOT="$PREPARED_DATA_ROOT" \
        RESULT_ROOT="$V2_ROOT/results/tseed${teacher_seed}" \
        POST_EVAL_ROOT="$V2_ROOT/post_eval/tseed${teacher_seed}" \
        EVAL_WORKERS=8 EVAL_PERSISTENT_WORKERS=1 \
        STUDENT_PROTOCOL_NAME=standard_protocol_v2 \
        STUDENT_INITIALIZATION=imagenet-v1 \
        STUDENT_BACKBONE_LR=1e-4 STUDENT_HEAD_LR=1e-3 \
        STUDENT_ADAMW_WEIGHT_DECAY=1e-5 \
        STUDENT_COSINE_T_MAX=400 STUDENT_COSINE_ETA_MIN=0 \
        STUDENT_TEMPERATURE=20 \
        bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" eval \
        "$dataset" "$recovery_seed" "$ipc" "$student_seed" > "$log" 2>&1
    python "$ROOT_DIR/fine_grained/record_standard_v2_result.py" \
        --result "$result" --reference-result "$reference" --dataset "$dataset" \
        --teacher-seed "$teacher_seed" --recovery-seed "$recovery_seed" \
        --ipc "$ipc" --student-seed "$student_seed" >> "$log" 2>&1
}

wait_batch() {
    local failed=0 pid
    for pid in "$@"; do wait "$pid" || failed=1; done
    (( failed == 0 ))
}

execute_task() {
    local dataset="$1" teacher_seed="$2" recovery_seed="$3"
    local task_id="${dataset}_t${teacher_seed}_r${recovery_seed}"
    local task_log="$LOG_ROOT/jobs/$dataset/tseed${teacher_seed}/rseed${recovery_seed}"
    local running="$STATUS_ROOT/${task_id}.running" complete="$STATUS_ROOT/${task_id}.complete"
    local failed="$STATUS_ROOT/${task_id}.failed" ipc student_seed pids=()
    mkdir -p "$task_log"; rm -f "$failed"
    echo "$(timestamp) gpu=$GPU_ID task=$task_id started" > "$running"
    cleanup() { local rc=$?; (( rc == 0 )) || echo "$(timestamp) exit=$rc" > "$failed"; rm -f "$running"; return "$rc"; }
    trap cleanup EXIT
    for ipc in "${IPCS[@]}"; do
        for student_seed in "${SEEDS[@]}"; do
            run_eval "$dataset" "$teacher_seed" "$recovery_seed" "$ipc" "$student_seed" "$task_log" &
            pids+=("$!")
            if (( ${#pids[@]} == EVAL_CONCURRENCY )); then wait_batch "${pids[@]}"; pids=(); fi
        done
    done
    if (( ${#pids[@]} )); then wait_batch "${pids[@]}"; fi
    echo "$(timestamp) gpu=$GPU_ID task=$task_id complete" > "$complete"
    rm -f "$running"; trap - EXIT
}

try_task() {
    local dataset="$1" teacher_seed="$2" recovery_seed="$3" task_id
    task_id="${dataset}_t${teacher_seed}_r${recovery_seed}"
    ( flock -n 7 || exit 75; [[ -f "$STATUS_ROOT/${task_id}.complete" ]] && exit 0; execute_task "$dataset" "$teacher_seed" "$recovery_seed" ) \
        7>"$LOCK_ROOT/task_${task_id}.lock"
}

worker_main() {
    [[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "worker requires numeric GPU" >&2; exit 2; }
    export CUDA_VISIBLE_DEVICES="$GPU_ID" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    export TORCH_HOME=/linxi/dataset/FD2/torch_cache LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
    while true; do
        local remaining=0 claimed=0 dataset teacher_seed recovery_seed rc
        for dataset in "${DATASETS[@]}"; do for teacher_seed in "${SEEDS[@]}"; do for recovery_seed in "${SEEDS[@]}"; do
            [[ -f "$STATUS_ROOT/${dataset}_t${teacher_seed}_r${recovery_seed}.complete" ]] && continue
            remaining=$((remaining + 1)); set +e; try_task "$dataset" "$teacher_seed" "$recovery_seed"; rc=$?; set -e
            if (( rc == 0 )); then claimed=1; break 3; fi
            if (( rc != 75 )); then return "$rc"; fi
        done; done; done
        (( remaining > 0 )) || return 0
        (( claimed == 1 )) || sleep 30
    done
}

launcher_main() {
    exec 9>"$LOCK_ROOT/launcher.lock"; flock -n 9 || { echo "v2 launcher already running" >&2; exit 1; }
    write_definition
    python "$ROOT_DIR/fine_grained/audit_standard_v2_inputs.py" \
        --v1-root "$V1_ROOT" --cub4k-root "$CUB4K_ROOT" \
        --output "$V2_ROOT/preflight/upstream_reuse_audit.json" \
        > "$LOG_ROOT/upstream_preflight.log" 2>&1
    echo "$(timestamp) launcher started" > "$STATUS_ROOT/launcher.running"
    local pids=() gpu failed=0 pid
    for gpu in $GPUS; do bash "$0" worker "$gpu" > "$LOG_ROOT/worker_gpu${gpu}.log" 2>&1 & pids+=("$!"); done
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    python "$ROOT_DIR/fine_grained/summarize_standard_v2_matrix.py" --v2-root "$V2_ROOT" > "$LOG_ROOT/summary.log" 2>&1 || failed=1
    rm -f "$STATUS_ROOT/launcher.running"
    if (( failed )); then echo "$(timestamp) launcher failed" > "$STATUS_ROOT/launcher.failed"; return 1; fi
    echo "$(timestamp) launcher complete" > "$STATUS_ROOT/launcher.complete"
}

case "$MODE" in
    prepare) write_definition ;;
    launch) launcher_main ;;
    worker) worker_main ;;
    *) echo "usage: $0 [prepare|launch|worker GPU]" >&2; exit 2 ;;
esac
