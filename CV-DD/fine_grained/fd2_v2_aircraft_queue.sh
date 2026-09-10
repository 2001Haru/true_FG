#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-launch}"
SEMANTICS="${2:-}"
GPU_ID="${3:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FD2_V1_ROOT="${FD2_V1_ROOT:-/linxi/dataset/FG_FD2_standard/v1}"
FD2_V2_ROOT="${FD2_V2_ROOT:-/linxi/dataset/FG_FD2_standard/v2}"
PLAIN_V2_ROOT="${PLAIN_V2_ROOT:-/linxi/dataset/FG_SRe2L_standard/v2}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-$FD2_V1_ROOT/datasets}"
CONTROL_ROOT="$FD2_V2_ROOT/aircraft_dual_semantics_control"
LOG_ROOT="$CONTROL_ROOT/logs"
STATUS_ROOT="$CONTROL_ROOT/status"
LOCK_ROOT="$CONTROL_ROOT/locks"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-4}"
IPCS=(1 3 5)
STUDENT_SEEDS=(42 43 44)
SEMANTICS_MODES=(released_semantics paper_literal)
mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$LOCK_ROOT"

timestamp() { date --iso-8601=seconds; }

write_definition() {
    local revision="$(git -C "$ROOT_DIR" rev-parse HEAD)"
    python - "$FD2_V2_ROOT" "$FD2_V1_ROOT" "$PLAIN_V2_ROOT" "$revision" <<'PY'
import json,os,sys
from pathlib import Path
v2,v1,plain=map(Path,sys.argv[1:4]); revision=sys.argv[4]
expected=[str((v2/m/"results/A_imsize224/rseed42"/f"ipc{i}_sseed{s}.json").resolve())
          for m in ("released_semantics","paper_literal") for i in (1,3,5) for s in (42,43,44)]
payload={"status":"prepared","experiment":"aircraft_fd2_dual_semantics_student_v2",
 "git_revision":revision,"semantics":["released_semantics","paper_literal"],"dataset":"A_imsize224",
 "teacher_seed":42,"recovery_seed":42,"ipcs":[1,3,5],"student_seeds":[42,43,44],
 "student_protocol":"fine_grained_sre2l_standard_protocol_v2","reuse":"existing semantic-specific synthetic images and FKD",
 "expected_results":18,"expected_result_files":expected,"fd2_v1_root":str(v1.resolve()),"plain_v2_root":str(plain.resolve())}
path=v2/"matrix_definition.json"; path.parent.mkdir(parents=True,exist_ok=True)
tmp=path.with_suffix(".json.tmp"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(tmp,path)
PY
}

preflight() {
    local semantics ipc seed old plain fkd
    [[ -d "$PREPARED_DATA_ROOT/A_imsize224/train" && -d "$PREPARED_DATA_ROOT/A_imsize224/test" ]]
    for semantics in "${SEMANTICS_MODES[@]}"; do
        [[ -f "$FD2_V1_ROOT/$semantics/teachers/A_imsize224/tseed42/ResNet18.pth" ]]
        [[ -f "$FD2_V1_ROOT/$semantics/recovery/A_imsize224/rseed42/recovery_manifest.json" ]]
        for ipc in "${IPCS[@]}"; do
            fkd="$FD2_V1_ROOT/$semantics/fkd/A_imsize224/rseed42/ipc${ipc}_bs20_ipc${ipc}"
            [[ -f "$fkd/relabel_manifest.json" && -f "$fkd/fkd_audit.json" ]]
            for seed in "${STUDENT_SEEDS[@]}"; do
                old="$FD2_V1_ROOT/$semantics/results/A_imsize224/rseed42/ipc${ipc}_sseed${seed}.json"
                plain="$PLAIN_V2_ROOT/results/tseed42/A_imsize224/rseed42/ipc${ipc}_sseed${seed}.json"
                [[ -f "$old" && -f "$plain" ]]
            done
        done
    done
}

run_eval() {
    local semantics="$1" ipc="$2" seed="$3" job_log="$4"
    local upstream="$FD2_V1_ROOT/$semantics"
    local result="$FD2_V2_ROOT/$semantics/results/A_imsize224/rseed42/ipc${ipc}_sseed${seed}.json"
    local old="$FD2_V1_ROOT/$semantics/results/A_imsize224/rseed42/ipc${ipc}_sseed${seed}.json"
    local teacher="$upstream/teachers/A_imsize224/tseed42"
    EXP_ROOT="$upstream" PREPARED_DATA_ROOT="$PREPARED_DATA_ROOT" \
        RESULT_ROOT="$FD2_V2_ROOT/$semantics/results" \
        POST_EVAL_ROOT="$FD2_V2_ROOT/$semantics/post_eval" \
        EVAL_WORKERS=8 EVAL_PERSISTENT_WORKERS=1 \
        STUDENT_PROTOCOL_NAME=standard_protocol_v2 STUDENT_INITIALIZATION=imagenet-v1 \
        STUDENT_BACKBONE_LR=1e-4 STUDENT_HEAD_LR=1e-3 STUDENT_ADAMW_WEIGHT_DECAY=1e-5 \
        STUDENT_COSINE_T_MAX=400 STUDENT_COSINE_ETA_MIN=0 STUDENT_TEMPERATURE=20 \
        bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" eval A_imsize224 42 "$ipc" "$seed" \
        > "$job_log/eval_ipc${ipc}_sseed${seed}.log" 2>&1
    python "$ROOT_DIR/fine_grained/annotate_fd2_result.py" --result "$result" --semantics "$semantics" \
        --joint-teacher "$teacher/FD2_ResNet18_CAL.pth" --backbone "$teacher/ResNet18.pth" \
        --recovery-manifest "$upstream/recovery/A_imsize224/rseed42/recovery_manifest.json" \
        >> "$job_log/eval_ipc${ipc}_sseed${seed}.log" 2>&1
    python "$ROOT_DIR/fine_grained/record_fd2_v2_result.py" --result "$result" \
        --reference-v1-result "$old" --semantics "$semantics" --ipc "$ipc" --student-seed "$seed" \
        >> "$job_log/eval_ipc${ipc}_sseed${seed}.log" 2>&1
}

wait_batch() { local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; (( failed==0 )); }

worker_main() {
    [[ "$SEMANTICS" == released_semantics || "$SEMANTICS" == paper_literal ]]
    [[ "$GPU_ID" =~ ^[01]$ ]]
    exec 8>"$LOCK_ROOT/${SEMANTICS}.lock"; flock -n 8 || exit 75
    local running="$STATUS_ROOT/${SEMANTICS}.running" complete="$STATUS_ROOT/${SEMANTICS}.complete"
    local failed="$STATUS_ROOT/${SEMANTICS}.failed" job_log="$LOG_ROOT/$SEMANTICS" ipc seed pids=()
    mkdir -p "$job_log"; rm -f "$failed"; echo "$(timestamp) started" > "$running"
    cleanup(){ local rc=$?; ((rc==0)) || echo "$(timestamp) exit=$rc" > "$failed"; rm -f "$running"; return "$rc"; }; trap cleanup EXIT
    export CUDA_VISIBLE_DEVICES="$GPU_ID" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    export TORCH_HOME=/linxi/dataset/FD2/torch_cache LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
    for ipc in "${IPCS[@]}"; do for seed in "${STUDENT_SEEDS[@]}"; do
        run_eval "$SEMANTICS" "$ipc" "$seed" "$job_log" & pids+=("$!")
        if (( ${#pids[@]}==EVAL_CONCURRENCY )); then
            if ! wait_batch "${pids[@]}"; then return 1; fi; pids=()
        fi
    done; done
    if (( ${#pids[@]} )); then if ! wait_batch "${pids[@]}"; then return 1; fi; fi
    echo "$(timestamp) complete" > "$complete"; rm -f "$running"; trap - EXIT
}

launcher_main() {
    exec 9>"$LOCK_ROOT/launcher.lock"; flock -n 9 || exit 75
    write_definition; preflight
    echo "$(timestamp) launcher started" > "$STATUS_ROOT/launcher.running"
    bash "$0" worker released_semantics 0 > "$LOG_ROOT/worker_gpu0.log" 2>&1 & local p0=$!
    bash "$0" worker paper_literal 1 > "$LOG_ROOT/worker_gpu1.log" 2>&1 & local p1=$!
    local failed=0; wait "$p0" || failed=1; wait "$p1" || failed=1
    rm -f "$STATUS_ROOT/launcher.running"
    if ((failed)); then echo "$(timestamp) failed" > "$STATUS_ROOT/launcher.failed"; return 1; fi
    python "$ROOT_DIR/fine_grained/summarize_fd2_v2_aircraft.py" --v2-root "$FD2_V2_ROOT" \
        --v1-root "$FD2_V1_ROOT" --plain-v2-root "$PLAIN_V2_ROOT" > "$LOG_ROOT/summary.log" 2>&1
    python - "$FD2_V2_ROOT/matrix_definition.json" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads(p.read_text()); x["status"]="complete"
t=p.with_suffix(".json.tmp"); t.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n"); os.replace(t,p)
PY
    echo "$(timestamp) complete" > "$STATUS_ROOT/launcher.complete"
}

case "$MODE" in launch) launcher_main;; worker) worker_main;; *) echo "usage: $0 [launch|worker SEMANTICS GPU]"; exit 2;; esac
