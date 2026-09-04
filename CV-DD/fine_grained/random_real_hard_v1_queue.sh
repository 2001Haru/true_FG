#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ROOT="${HARD_LABEL_V1_EXP_ROOT:-/linxi/dataset/FG_HardLabel_standard/v1/random_real}"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
LOG_ROOT="$EXP_ROOT/logs"
STATUS_PATH="$EXP_ROOT/queue_status.json"
MATRIX_PATH="$EXP_ROOT/matrix_definition.json"
GLOBAL_LOCK_ROOT="${GLOBAL_GPU_LOCK_ROOT:-/linxi/dataset/FG_CoDA_standard/v2/locks}"
DATASETS=(CUB_imsize224 A_imsize224 SC_imsize224)
IPCS=(1 3 5)
SELECTION_SEEDS=(0 1 2)
STUDENT_SEEDS=(42 43 44)
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

write_definition() {
    local revision
    revision="$(git -C "$ROOT_DIR" rev-parse HEAD)"
    python - "$MATRIX_PATH" "$EXP_ROOT" "$revision" "$GPU_COUNT" "$EVALS_PER_GPU" \
        "$WAVE_SIZE" "$MEMORY_LIMIT_BYTES" <<'PY'
import json, os, sys
from pathlib import Path
path=Path(sys.argv[1]); root=Path(sys.argv[2])
payload={
 'status':'running','protocol':'hard_label_v1','method':'RandomReal',
 'git_revision':sys.argv[3], 'datasets':['CUB_imsize224','A_imsize224','SC_imsize224'],
 'ipcs':[1,3,5], 'selection_seeds':[0,1,2], 'student_seeds':[42,43,44],
 'expected_selected_sets':27, 'expected_results':81,
 'student':'torchvision ResNet18, ImageNet1K V1 backbone, random head',
 'optimizer':'SGD', 'backbone_lr':3e-4, 'head_lr':3e-3,
 'minimum_lrs':{'backbone':0.0,'head':0.0}, 'momentum':0.9, 'weight_decay':5e-4,
 'weight_decay_excludes':'BN and bias', 'scheduler':'per-update full cosine',
 'total_optimizer_updates':3000, 'physical_batch_size':64,
 'gradient_accumulation_steps':1, 'bn_running_statistics':'updated',
 'cutmix':False, 'normalization':'ImageNet mean/std',
 'train_transform':'Resize256->RandomCrop224->HorizontalFlip',
 'test_transform':'Resize256->CenterCrop224', 'primary_metric':'final_update_top1',
 'gpu_count':int(sys.argv[4]), 'max_eval_concurrency_per_gpu':int(sys.argv[5]),
 'eval_wave_size':int(sys.argv[6]), 'cgroup_memory_limit_bytes':int(sys.argv[7]),
 'eval_openblas_num_threads':1,
 'expected_result_files':[
  str((root/'results'/d/f'ipc{i}_rseed{r}_sseed{s}.json').resolve())
  for d in ('CUB_imsize224','A_imsize224','SC_imsize224')
  for i in (1,3,5) for r in (0,1,2) for s in (42,43,44)
 ],
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
write_status running prepare
for dataset in "${DATASETS[@]}"; do
    for selection_seed in "${SELECTION_SEEDS[@]}"; do
        for ipc in "${IPCS[@]}"; do
            CURRENT_STAGE="prepare:$dataset:ipc$ipc:rseed$selection_seed"
            mkdir -p "$LOG_ROOT/$dataset/ipc${ipc}/rseed${selection_seed}"
            DATA_ROOT="$DATA_ROOT" HARD_LABEL_V1_EXP_ROOT="$EXP_ROOT" \
                bash "$ROOT_DIR/CV-DD/fine_grained/run_random_real_hard_v1.sh" \
                prepare "$dataset" "$ipc" "$selection_seed" \
                > "$LOG_ROOT/$dataset/ipc${ipc}/rseed${selection_seed}/prepare.log" 2>&1
        done
    done
done

wait_wave() {
    local failed=0 pid
    for pid in "$@"; do wait "$pid" || failed=1; done
    (( failed == 0 ))
}

run_eval() {
    local dataset="$1" ipc="$2" selection_seed="$3" student_seed="$4" gpu="$5"
    CUDA_VISIBLE_DEVICES="$gpu" OPENBLAS_NUM_THREADS=1 DATA_ROOT="$DATA_ROOT" \
        HARD_LABEL_V1_EXP_ROOT="$EXP_ROOT" \
        bash "$ROOT_DIR/CV-DD/fine_grained/run_random_real_hard_v1.sh" \
        eval "$dataset" "$ipc" "$selection_seed" "$student_seed" \
        > "$LOG_ROOT/$dataset/ipc${ipc}/rseed${selection_seed}/eval_sseed${student_seed}.log" 2>&1
}

CURRENT_STAGE=evaluation
write_status running evaluation
task_index=0
pids=()
for dataset in "${DATASETS[@]}"; do
    for ipc in "${IPCS[@]}"; do
        for selection_seed in "${SELECTION_SEEDS[@]}"; do
            for student_seed in "${STUDENT_SEEDS[@]}"; do
                gpu=$((task_index % GPU_COUNT))
                run_eval "$dataset" "$ipc" "$selection_seed" "$student_seed" "$gpu" &
                pids+=("$!")
                task_index=$((task_index + 1))
                if (( ${#pids[@]} == WAVE_SIZE )); then
                    wait_wave "${pids[@]}"
                    pids=()
                fi
            done
        done
    done
done
if (( ${#pids[@]} > 0 )); then wait_wave "${pids[@]}"; fi

CURRENT_STAGE=summary
python "$ROOT_DIR/CV-DD/fine_grained/summarize_random_real_hard_v1.py" \
    --experiment-root "$EXP_ROOT" --output "$EXP_ROOT/summary/random_real_hard_v1.json" \
    > "$LOG_ROOT/summary.log" 2>&1
set_matrix_status complete
write_status complete complete
trap - EXIT

