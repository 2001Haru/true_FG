#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ROOT="${DINO_FIVEARM_EXP_ROOT:-/linxi/dataset/FG_HardLabel_standard/v1/dino_fivearm_ipc1}"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
LOG_ROOT="$EXP_ROOT/logs"
STATUS_PATH="$EXP_ROOT/queue_status.json"
MATRIX_PATH="$EXP_ROOT/matrix_definition.json"
GLOBAL_LOCK_ROOT="${GLOBAL_GPU_LOCK_ROOT:-/linxi/dataset/FG_CoDA_standard/v2/locks}"
DATASETS=(CUB_imsize224 A_imsize224 SC_imsize224)
DETERMINISTIC_ARMS=(centroid rival_facing_edge outward_edge edge_high_margin)
DETERMINISTIC_STUDENT_SEEDS=(42 43 44 45 46 47)
RANDOM_ARMS=(random_rseed0 random_rseed1 random_rseed2)
RANDOM_STUDENT_SEEDS=(42 43 44)
GPU_COUNT="${HARD_LABEL_V1_GPU_COUNT:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}"
EVALS_PER_GPU="${HARD_LABEL_V1_EVALS_PER_GPU:-2}"
WAVE_SIZE="${HARD_LABEL_V1_WAVE_SIZE:-$((GPU_COUNT * EVALS_PER_GPU))}"
MEMORY_PER_EVAL_GIB="${HARD_LABEL_V1_MEMORY_GIB_PER_EVAL:-16}"
MEMORY_HEADROOM_GIB="${HARD_LABEL_V1_MEMORY_HEADROOM_GIB:-8}"
MEMORY_LIMIT_BYTES="$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo 0)"
MEMORY_REQUIRED_BYTES=$(((WAVE_SIZE * MEMORY_PER_EVAL_GIB + MEMORY_HEADROOM_GIB) * 1024 * 1024 * 1024))
CURRENT_STAGE=initializing

[[ "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "invalid GPU count" >&2; exit 2; }
[[ "$EVALS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || { echo "invalid evals/GPU" >&2; exit 2; }
if [[ "$MEMORY_LIMIT_BYTES" =~ ^[0-9]+$ ]] \
    && (( MEMORY_LIMIT_BYTES > 0 && MEMORY_LIMIT_BYTES < MEMORY_REQUIRED_BYTES )); then
    echo "Unsafe hard-label v1 wave: cgroup memory below $MEMORY_REQUIRED_BYTES" >&2
    exit 2
fi
mkdir -p "$EXP_ROOT" "$LOG_ROOT" "$EXP_ROOT/summary" "$GLOBAL_LOCK_ROOT"

write_status() {
    python - "$STATUS_PATH" "$1" "$2" "${3:-0}" <<'PY'
import datetime, json, os, sys
from pathlib import Path
path=Path(sys.argv[1]); payload={
 'status':sys.argv[2], 'stage':sys.argv[3], 'exit_code':int(sys.argv[4]),
 'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
tmp=path.with_suffix(path.suffix+'.tmp')
tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
os.replace(tmp,path)
PY
}

set_matrix_status() {
    python - "$MATRIX_PATH" "$1" <<'PY'
import json, os, sys
from pathlib import Path
path=Path(sys.argv[1]); payload=json.loads(path.read_text(encoding='utf-8'))
payload['status']=sys.argv[2]; tmp=path.with_suffix(path.suffix+'.tmp')
tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
os.replace(tmp,path)
PY
}

write_definition() {
    local revision protocol_sha
    revision="$(git -C "$ROOT_DIR" rev-parse HEAD)"
    protocol_sha="$(sha256sum "$ROOT_DIR/CV-DD/fine_grained/hard_label_v1_protocol.json" | awk '{print $1}')"
    python - "$MATRIX_PATH" "$EXP_ROOT" "$revision" "$protocol_sha" "$GPU_COUNT" \
        "$EVALS_PER_GPU" "$WAVE_SIZE" "$MEMORY_LIMIT_BYTES" <<'PY'
import json, os, sys
from pathlib import Path
path=Path(sys.argv[1]); root=Path(sys.argv[2])
datasets=('CUB_imsize224','A_imsize224','SC_imsize224')
det=('centroid','rival_facing_edge','outward_edge','edge_high_margin')
random=('random_rseed0','random_rseed1','random_rseed2')
expected=[]
for dataset in datasets:
 for arm in det:
  expected.extend(str((root/'results'/dataset/arm/f'sseed{s}.json').resolve()) for s in range(42,48))
 for arm in random:
  expected.extend(str((root/'results'/dataset/arm/f'sseed{s}.json').resolve()) for s in range(42,45))
payload={
 'status':'running','experiment':'dino_fivearm_ipc1','protocol':'hard_label_v1',
 'git_revision':sys.argv[3], 'protocol_spec_sha256':sys.argv[4],
 'datasets':list(datasets),'ipc':1,
 'deterministic_selection_arms':list(det),
 'deterministic_student_seeds':list(range(42,48)),
 'random_selection_arms':list(random),'random_selection_seeds':[0,1,2],
 'random_student_seeds':[42,43,44], 'expected_results':99,
 'selection_geometry':'DINOv2 L2-normalized pooler output; normalized class prototypes; cosine',
 'edge_shell':{
  'prototype_correctness_filter':False,
  'radial_percentile':[0.70,0.95],
  'revision_reason':'m>=0 tests an unadapted DINO nearest-centroid proxy, not label validity',
 },
 'post_eval_image_augmentation':'hard_label_v1 unchanged: Resize256->RandomCrop224->flip',
 'selection_image_augmentation':'none: deterministic Resize256->CenterCrop224',
 'gpu_count':int(sys.argv[5]),'max_eval_concurrency_per_gpu':int(sys.argv[6]),
 'eval_wave_size':int(sys.argv[7]),'cgroup_memory_limit_bytes':int(sys.argv[8]),
 'eval_openblas_num_threads':1,'expected_result_files':expected,
}
tmp=path.with_suffix(path.suffix+'.tmp')
tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
os.replace(tmp,path)
PY
}

on_exit() {
    local exit_code=$?
    if (( exit_code != 0 )); then
        write_status failed "$CURRENT_STAGE" "$exit_code"
        [[ -f "$MATRIX_PATH" ]] && set_matrix_status failed
    fi
}
trap on_exit EXIT

exec 9>"$GLOBAL_LOCK_ROOT/launcher.lock"
flock -n 9 || { echo "Another full GPU queue is running" >&2; exit 1; }
write_definition
write_status running selection
for dataset in "${DATASETS[@]}"; do
    CURRENT_STAGE="selection:$dataset"
    mkdir -p "$LOG_ROOT/$dataset"
    DATA_ROOT="$DATA_ROOT" DINO_FIVEARM_EXP_ROOT="$EXP_ROOT" \
        bash "$ROOT_DIR/CV-DD/fine_grained/run_dino_fourway_hard_v1.sh" \
        prepare "$dataset" > "$LOG_ROOT/$dataset/selection.log" 2>&1
done

wait_wave() {
    local failed=0 pid
    for pid in "$@"; do wait "$pid" || failed=1; done
    (( failed == 0 ))
}

run_eval() {
    local dataset="$1" arm="$2" student_seed="$3" gpu="$4"
    mkdir -p "$LOG_ROOT/$dataset/$arm"
    CUDA_VISIBLE_DEVICES="$gpu" OPENBLAS_NUM_THREADS=1 DATA_ROOT="$DATA_ROOT" \
        DINO_FIVEARM_EXP_ROOT="$EXP_ROOT" \
        bash "$ROOT_DIR/CV-DD/fine_grained/run_dino_fourway_hard_v1.sh" \
        eval "$dataset" "$arm" "$student_seed" \
        > "$LOG_ROOT/$dataset/$arm/eval_sseed${student_seed}.log" 2>&1
}

launch_task() {
    local dataset="$1" arm="$2" student_seed="$3" gpu
    gpu=$((task_index % GPU_COUNT))
    run_eval "$dataset" "$arm" "$student_seed" "$gpu" &
    pids+=("$!")
    task_index=$((task_index + 1))
    if (( ${#pids[@]} == WAVE_SIZE )); then
        wait_wave "${pids[@]}"
        pids=()
    fi
}

CURRENT_STAGE=evaluation
write_status running evaluation
task_index=0
pids=()
for dataset in "${DATASETS[@]}"; do
    for arm in "${DETERMINISTIC_ARMS[@]}"; do
        for student_seed in "${DETERMINISTIC_STUDENT_SEEDS[@]}"; do
            launch_task "$dataset" "$arm" "$student_seed"
        done
    done
    for arm in "${RANDOM_ARMS[@]}"; do
        for student_seed in "${RANDOM_STUDENT_SEEDS[@]}"; do
            launch_task "$dataset" "$arm" "$student_seed"
        done
    done
done
if (( ${#pids[@]} > 0 )); then wait_wave "${pids[@]}"; fi

CURRENT_STAGE=summary
python "$ROOT_DIR/CV-DD/fine_grained/summarize_dino_fourway_hard_v1.py" \
    --experiment-root "$EXP_ROOT" --output "$EXP_ROOT/summary/dino_fivearm_hard_v1.json" \
    > "$LOG_ROOT/summary.log" 2>&1
set_matrix_status complete
write_status complete complete
trap - EXIT
