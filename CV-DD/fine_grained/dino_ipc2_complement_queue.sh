#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ROOT="${DINO_IPC2_EXP_ROOT:-/linxi/dataset/FG_HardLabel_standard/v1/dino_ipc2_center_complement}"
PARENT_ROOT="${DINO_IPC1_PARENT_ROOT:-/linxi/dataset/FG_HardLabel_standard/v1/dino_fivearm_ipc1}"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
LOG_ROOT="$EXP_ROOT/logs"
STATUS_PATH="$EXP_ROOT/queue_status.json"
MATRIX_PATH="$EXP_ROOT/matrix_definition.json"
GLOBAL_LOCK_ROOT="${GLOBAL_GPU_LOCK_ROOT:-/linxi/dataset/FG_CoDA_standard/v2/locks}"
DATASETS=(CUB_imsize224 A_imsize224 SC_imsize224)
STOCHASTIC=(random_ipc2 spherical_kmeans2 center_plus_random center_plus_shell_random)
DETERMINISTIC=(center_plus_outward center_plus_high_margin center_plus_rival_facing global_center_top2)
GPU_COUNT="${HARD_LABEL_V1_GPU_COUNT:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}"
EVALS_PER_GPU="${HARD_LABEL_V1_EVALS_PER_GPU:-2}"
WAVE_SIZE="${HARD_LABEL_V1_WAVE_SIZE:-$((GPU_COUNT * EVALS_PER_GPU))}"
MEMORY_PER_EVAL_GIB="${HARD_LABEL_V1_MEMORY_GIB_PER_EVAL:-16}"
MEMORY_HEADROOM_GIB="${HARD_LABEL_V1_MEMORY_HEADROOM_GIB:-8}"
MEMORY_LIMIT_BYTES="$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo 0)"
MEMORY_REQUIRED_BYTES=$(((WAVE_SIZE * MEMORY_PER_EVAL_GIB + MEMORY_HEADROOM_GIB) * 1024 * 1024 * 1024))
CURRENT_STAGE=initializing

[[ "$GPU_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "invalid GPU count" >&2; exit 2; }
if [[ "$MEMORY_LIMIT_BYTES" =~ ^[0-9]+$ ]] \
    && (( MEMORY_LIMIT_BYTES > 0 && MEMORY_LIMIT_BYTES < MEMORY_REQUIRED_BYTES )); then
    echo "Unsafe IPC2 wave: cgroup memory below $MEMORY_REQUIRED_BYTES" >&2
    exit 2
fi
mkdir -p "$EXP_ROOT" "$LOG_ROOT" "$EXP_ROOT/summary" "$GLOBAL_LOCK_ROOT"

write_status() {
    python - "$STATUS_PATH" "$1" "$2" "${3:-0}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); x={'status':sys.argv[2],'stage':sys.argv[3],'exit_code':int(sys.argv[4]),'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}; t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
}
set_matrix_status() {
    python - "$MATRIX_PATH" "$1" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads(p.read_text()); x['status']=sys.argv[2]; t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
}
write_definition() {
    local revision
    revision="$(git -C "$ROOT_DIR" rev-parse HEAD)"
    python - "$MATRIX_PATH" "$EXP_ROOT" "$revision" "$GPU_COUNT" "$EVALS_PER_GPU" "$WAVE_SIZE" "$MEMORY_LIMIT_BYTES" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); root=Path(sys.argv[2]); datasets=('CUB_imsize224','A_imsize224','SC_imsize224'); stochastic=('random_ipc2','spherical_kmeans2','center_plus_random','center_plus_shell_random'); deterministic=('center_plus_outward','center_plus_high_margin','center_plus_rival_facing','global_center_top2'); expected=[]
for d in datasets:
 for method in stochastic:
  for r in (0,1,2): expected.extend(str((root/'results'/d/f'{method}_rseed{r}'/f'sseed{s}.json').resolve()) for s in (42,43,44))
 for arm in deterministic: expected.extend(str((root/'results'/d/arm/f'sseed{s}.json').resolve()) for s in range(42,48))
x={'status':'running','experiment':'dino_ipc2_center_complement','protocol':'hard_label_v1','git_revision':sys.argv[3],'datasets':list(datasets),'ipc':2,'stochastic_methods':list(stochastic),'selection_seeds':[0,1,2],'stochastic_student_seeds':[42,43,44],'deterministic_methods':list(deterministic),'deterministic_student_seeds':list(range(42,48)),'expected_results':180,'total_optimizer_updates':3000,'post_eval_protocol_changed':False,'gpu_count':int(sys.argv[4]),'max_eval_concurrency_per_gpu':int(sys.argv[5]),'eval_wave_size':int(sys.argv[6]),'cgroup_memory_limit_bytes':int(sys.argv[7]),'expected_result_files':expected}; t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
}
on_exit() { local code=$?; if (( code != 0 )); then write_status failed "$CURRENT_STAGE" "$code"; [[ -f "$MATRIX_PATH" ]] && set_matrix_status failed; fi; }
trap on_exit EXIT
exec 9>"$GLOBAL_LOCK_ROOT/launcher.lock"
flock -n 9 || { echo "Another full GPU queue is running" >&2; exit 1; }
write_definition
write_status running selection
for dataset in "${DATASETS[@]}"; do
    CURRENT_STAGE="selection:$dataset"; mkdir -p "$LOG_ROOT/$dataset"
    DATA_ROOT="$DATA_ROOT" DINO_IPC1_PARENT_ROOT="$PARENT_ROOT" DINO_IPC2_EXP_ROOT="$EXP_ROOT" \
      bash "$ROOT_DIR/CV-DD/fine_grained/run_dino_ipc2_complement.sh" prepare "$dataset" >"$LOG_ROOT/$dataset/selection.log" 2>&1
done
wait_wave() { local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; (( failed == 0 )); }
run_eval() {
    local dataset="$1" arm="$2" seed="$3" gpu="$4"; mkdir -p "$LOG_ROOT/$dataset/$arm"
    CUDA_VISIBLE_DEVICES="$gpu" OPENBLAS_NUM_THREADS=1 DATA_ROOT="$DATA_ROOT" DINO_IPC1_PARENT_ROOT="$PARENT_ROOT" DINO_IPC2_EXP_ROOT="$EXP_ROOT" \
      bash "$ROOT_DIR/CV-DD/fine_grained/run_dino_ipc2_complement.sh" eval "$dataset" "$arm" "$seed" >"$LOG_ROOT/$dataset/$arm/eval_sseed${seed}.log" 2>&1
}
launch() { local dataset="$1" arm="$2" seed="$3" gpu=$((task_index % GPU_COUNT)); run_eval "$dataset" "$arm" "$seed" "$gpu" & pids+=("$!"); task_index=$((task_index+1)); if (( ${#pids[@]} == WAVE_SIZE )); then wait_wave "${pids[@]}"; pids=(); fi; }
CURRENT_STAGE=evaluation; write_status running evaluation; task_index=0; pids=()
for dataset in "${DATASETS[@]}"; do
  for method in "${STOCHASTIC[@]}"; do for rseed in 0 1 2; do for sseed in 42 43 44; do launch "$dataset" "${method}_rseed${rseed}" "$sseed"; done; done; done
  for arm in "${DETERMINISTIC[@]}"; do for sseed in 42 43 44 45 46 47; do launch "$dataset" "$arm" "$sseed"; done; done
done
if (( ${#pids[@]} > 0 )); then wait_wave "${pids[@]}"; fi
CURRENT_STAGE=summary
python "$ROOT_DIR/CV-DD/fine_grained/summarize_dino_ipc2_complement.py" --experiment-root "$EXP_ROOT" --parent-ipc1-summary "$PARENT_ROOT/summary/dino_sixarm_hard_v1.json" --output "$EXP_ROOT/summary/dino_ipc2_center_complement.json" >"$LOG_ROOT/summary.log" 2>&1
set_matrix_status complete; write_status complete complete; trap - EXIT

