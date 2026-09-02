#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-launch}"
SEMANTICS="${2:-}"
GPU_ID="${3:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FD2_ROOT="${FD2_ROOT:-/linxi/dataset/FG_FD2_standard/v1}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
CONTROL_ROOT="$FD2_ROOT/aircraft_pilot_control"
LOG_ROOT="$CONTROL_ROOT/logs"
STATUS_ROOT="$CONTROL_ROOT/status"
LOCK_ROOT="$CONTROL_ROOT/locks"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-4}"
DATASET=A_imsize224
TEACHER_SEED=42
RECOVERY_SEED=42
IPCS=(1 3 5)
STUDENT_SEEDS=(42 43 44)
SEMANTIC_MODES=(released_semantics paper_literal)
mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$LOCK_ROOT"

timestamp() { date --iso-8601=seconds; }

write_definition() {
    local revision
    revision="$(git -C "$ROOT_DIR" rev-parse HEAD)"
    python - "$FD2_ROOT" "$CONTROL_ROOT" "$revision" "$EVAL_CONCURRENCY" <<'PY'
import json
import os
import sys
from pathlib import Path

root, control = map(Path, sys.argv[1:3])
revision, concurrency = sys.argv[3], int(sys.argv[4])
semantics = ["released_semantics", "paper_literal"]
expected = []
for mode in semantics:
    for ipc in (1, 3, 5):
        for student_seed in (42, 43, 44):
            expected.append(str((
                root / mode / "results" / "A_imsize224" / "rseed42" /
                f"ipc{ipc}_sseed{student_seed}.json"
            ).resolve()))
payload = {
    "status": "running",
    "experiment": "aircraft_fd2_dual_semantics_pilot",
    "git_revision": revision,
    "dataset": "A_imsize224",
    "semantics": semantics,
    "teacher_seed": 42,
    "recovery_seed": 42,
    "student_seeds": [42, 43, 44],
    "ipcs": [1, 3, 5],
    "recovery_iterations": 4000,
    "expected_results_per_semantics": 9,
    "expected_results": 18,
    "eval_concurrency_per_gpu": concurrency,
    "gpu_assignment": {"released_semantics": 0, "paper_literal": 1},
    "expected_result_files": expected,
    "log_root": str((control / "logs").resolve()),
}
path = control / "pilot_definition.json"
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

wait_batch() {
    local failed=0 pid
    for pid in "$@"; do wait "$pid" || failed=1; done
    (( failed == 0 ))
}

run_eval() {
    local semantics="$1" ipc="$2" student_seed="$3" job_log="$4"
    local result="$FD2_ROOT/$semantics/results/$DATASET/rseed${RECOVERY_SEED}/ipc${ipc}_sseed${student_seed}.json"
    bash "$ROOT_DIR/fine_grained/run_fd2_standard_fg.sh" eval "$semantics" \
        "$DATASET" "$RECOVERY_SEED" "$ipc" "$student_seed" \
        > "$job_log/eval_ipc${ipc}_sseed${student_seed}.log" 2>&1
    python "$ROOT_DIR/fine_grained/audit_result.py" \
        --result "$result" --classes 100 --validation-images 3333 \
        >> "$job_log/eval_ipc${ipc}_sseed${student_seed}.log" 2>&1
}

worker_main() {
    [[ "$SEMANTICS" == released_semantics || "$SEMANTICS" == paper_literal ]] || {
        echo "worker semantics must be released_semantics or paper_literal" >&2; exit 2;
    }
    [[ "$GPU_ID" =~ ^[01]$ ]] || { echo "worker requires GPU 0 or 1" >&2; exit 2; }
    local lock="$LOCK_ROOT/${SEMANTICS}.lock"
    exec 8>"$lock"
    flock -n 8 || { echo "$SEMANTICS worker already running" >&2; exit 75; }
    local running="$STATUS_ROOT/${SEMANTICS}.running"
    local complete="$STATUS_ROOT/${SEMANTICS}.complete"
    local failed="$STATUS_ROOT/${SEMANTICS}.failed"
    local job_log="$LOG_ROOT/$SEMANTICS"
    mkdir -p "$job_log"
    echo "$(timestamp) gpu=$GPU_ID semantics=$SEMANTICS started" > "$running"
    rm -f "$failed"
    cleanup() {
        local rc=$?
        if (( rc != 0 )); then echo "$(timestamp) exit=$rc" > "$failed"; fi
        rm -f "$running"
        return "$rc"
    }
    trap cleanup EXIT
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    export TORCH_HOME=/linxi/dataset/FD2/torch_cache
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
    export FD2_ROOT PREPARED_DATA_ROOT
    export TEACHER_SEED PATCH_SEED=42
    export RELABEL_MANIFEST_REQUIRED=1
    export STUDENT_INITIALIZATION=random
    export STUDENT_ADAMW_LR=1e-3
    export STUDENT_ADAMW_WEIGHT_DECAY=1e-5
    export STUDENT_ETA=2
    export STUDENT_TEMPERATURE=20

    bash "$ROOT_DIR/fine_grained/run_fd2_standard_fg.sh" teacher "$SEMANTICS" \
        "$DATASET" "$TEACHER_SEED" > "$job_log/teacher.log" 2>&1
    bash "$ROOT_DIR/fine_grained/run_fd2_standard_fg.sh" patches "$SEMANTICS" \
        "$DATASET" "$RECOVERY_SEED" > "$job_log/patches.log" 2>&1
    bash "$ROOT_DIR/fine_grained/run_fd2_standard_fg.sh" recover "$SEMANTICS" \
        "$DATASET" "$RECOVERY_SEED" > "$job_log/recover.log" 2>&1
    bash "$ROOT_DIR/fine_grained/run_fd2_standard_fg.sh" sample "$SEMANTICS" \
        "$DATASET" "$RECOVERY_SEED" > "$job_log/sample.log" 2>&1
    local ipc student_seed pids=()
    for ipc in "${IPCS[@]}"; do
        bash "$ROOT_DIR/fine_grained/run_fd2_standard_fg.sh" relabel "$SEMANTICS" \
            "$DATASET" "$RECOVERY_SEED" "$ipc" > "$job_log/relabel_ipc${ipc}.log" 2>&1
    done
    for ipc in "${IPCS[@]}"; do
        for student_seed in "${STUDENT_SEEDS[@]}"; do
            run_eval "$SEMANTICS" "$ipc" "$student_seed" "$job_log" &
            pids+=("$!")
            if (( ${#pids[@]} == EVAL_CONCURRENCY )); then
                wait_batch "${pids[@]}"
                pids=()
            fi
        done
    done
    if (( ${#pids[@]} > 0 )); then wait_batch "${pids[@]}"; fi
    echo "$(timestamp) gpu=$GPU_ID semantics=$SEMANTICS complete" > "$complete"
    rm -f "$running"
    trap - EXIT
}

launcher_main() {
    command -v flock >/dev/null 2>&1 || { echo "flock is required" >&2; exit 1; }
    exec 9>"$LOCK_ROOT/launcher.lock"
    flock -n 9 || { echo "Aircraft FD2 pilot launcher is already running" >&2; exit 1; }
    [[ -d "$PREPARED_DATA_ROOT/$DATASET/train" && -d "$PREPARED_DATA_ROOT/$DATASET/test" ]] || {
        echo "missing prepared dataset: $PREPARED_DATA_ROOT/$DATASET" >&2; exit 1;
    }
    write_definition
    FD2_ROOT="$FD2_ROOT" PREPARED_DATA_ROOT="$PREPARED_DATA_ROOT" \
        bash "$ROOT_DIR/fine_grained/run_fd2_standard_fg.sh" audit released_semantics \
        "$DATASET" 42 > "$LOG_ROOT/input_audit.log" 2>&1
    echo "$(timestamp) launcher started" > "$STATUS_ROOT/launcher.running"
    bash "$0" worker released_semantics 0 > "$LOG_ROOT/worker_gpu0.log" 2>&1 &
    local pid0=$!
    bash "$0" worker paper_literal 1 > "$LOG_ROOT/worker_gpu1.log" 2>&1 &
    local pid1=$!
    local failed=0
    wait "$pid0" || failed=1
    wait "$pid1" || failed=1
    rm -f "$STATUS_ROOT/launcher.running"
    if (( failed )); then
        echo "$(timestamp) launcher failed" > "$STATUS_ROOT/launcher.failed"
        return 1
    fi
    python - "$CONTROL_ROOT/pilot_definition.json" <<'PY'
import json
import os
import sys
from pathlib import Path
path = Path(sys.argv[1]); payload = json.loads(path.read_text(encoding="utf-8"))
missing = [item for item in payload["expected_result_files"] if not Path(item).is_file()]
if missing:
    raise RuntimeError(f"missing {len(missing)} expected results")
payload["status"] = "complete"
temporary = path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
    echo "$(timestamp) launcher complete" > "$STATUS_ROOT/launcher.complete"
}

case "$MODE" in
    launch) launcher_main ;;
    worker) worker_main ;;
    *) echo "usage: $0 [launch|worker SEMANTICS GPU]" >&2; exit 2 ;;
esac
