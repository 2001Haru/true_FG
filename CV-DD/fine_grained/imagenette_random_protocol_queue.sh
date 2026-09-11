#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_ImageNette_full_RDED_protocol/v1}"
AB_ROOT="${AB_ROOT:-/linxi/dataset/FG_ImageNette_native_resolution/v1}"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/CV-DD/test_data/imagenet-nette}"
TEACHER="${TEACHER:-/linxi/dataset/CV-DD/offline_models/imagenet-nette/ResNet18.pth}"
RANDOM_ROOT="$EXP_ROOT/random_selection"
GPUS=(${GPUS:-0 1}); IPCS=(10 50); STUDENT_SEEDS=(42 43 44)
LOG_ROOT="$EXP_ROOT/logs/random"; STATUS_ROOT="$EXP_ROOT/status/random"; LOCK_ROOT="$EXP_ROOT/locks"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$LOCK_ROOT" "$EXP_ROOT/summary"

timestamp(){ date --iso-8601=seconds; }
wait_all(){ local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; ((failed==0)); }
images(){ echo "$RANDOM_ROOT/ipc$1"; }
manifest(){ echo "$RANDOM_ROOT/ipc$1/selection_manifest.json"; }
fkd(){ echo "$EXP_ROOT/fkd/current_cvdd/random_real/ipc${1}_bs10_ipc${1}"; }

prepare(){
  local ipc
  for ipc in "${IPCS[@]}"; do
    python "$ROOT_DIR/CV-DD/fine_grained/prepare_random_real_fg.py" --data-dir "$DATA_ROOT" \
      --output-dir "$(images "$ipc")" --manifest "$(manifest "$ipc")" \
      --dataset-name imagenet-nette --classes 10 --ipc "$ipc" --selection-seed 42 --link-mode symlink \
      > "$LOG_ROOT/selection_ipc${ipc}.log" 2>&1
  done
  python - "$RANDOM_ROOT" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]); a=json.load(open(r/'ipc10/selection_manifest.json')); b=json.load(open(r/'ipc50/selection_manifest.json'))
sa={(x['class_id'],x['source_sha256']) for x in a['images']}; sb={(x['class_id'],x['source_sha256']) for x in b['images']}
if len(sa)!=100 or len(sb)!=500 or not sa <= sb: raise SystemExit('IPC10 is not a strict subset of IPC50')
print('nested_selection_audit=complete overlap=100/100')
PY
}

relabel_one(){
  local ipc="$1" gpu="$2" base="$EXP_ROOT/fkd/current_cvdd/random_real/ipc${ipc}" actual expected count=0
  actual="$(fkd "$ipc")"; expected=$((300*ipc))
  [[ -d "$actual" ]] && count="$(find "$actual" -type f -name 'batch_*.tar' | wc -l)"
  if (( count != expected )); then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      python -u "$ROOT_DIR/CV-DD/relabel/relabel.py" --syn-data-path "$(images "$ipc")" \
        --fkd-path "$base" --model-pool-dir "$(dirname "$TEACHER")" --teacher-model-name ResNet18 \
        --gpu 0 --batch-size 10 --workers 2 --dataset-name imagenet-nette --epochs 300 \
        --seed 42 --fkd-seed 42 --min-scale-crops 0.08 --max-scale-crops 1 \
        --use-fp16 --mode fkd_save --mix-type cutmix > "$LOG_ROOT/relabel_ipc${ipc}.log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$actual" \
    --images $((10*ipc)) --classes 10 --batch-size 10 --epochs 300 >> "$LOG_ROOT/relabel_ipc${ipc}.log" 2>&1
}

current_one(){
  local ipc="$1" seed="$2" gpu="$3" eta result log
  [[ "$ipc" == 10 ]] && eta=1 || eta=2
  result="$EXP_ROOT/results/current_cvdd/random_real/ipc${ipc}_sseed${seed}.json"
  log="$LOG_ROOT/current_ipc${ipc}_sseed${seed}.log"; mkdir -p "$(dirname "$result")"
  exec 7>"${result}.lock"; flock -n 7 || return 75
  if [[ ! -f "$result" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -u "$ROOT_DIR/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc "$ipc" \
        --exp-name "imagenette_random_current_ipc${ipc}_s${seed}" --original-data-path "$(images "$ipc")" \
        --fkd-path "$(fkd "$ipc")" --output-dir "$EXP_ROOT/post_eval" --batch-size 10 --epochs 300 \
        --dataset-name imagenet-nette --gradient-accumulation-steps 2 --mix-type cutmix --workers 2 \
        --fkd_seed 42 --train-seed "$seed" --temperature 20 --student-initialization random \
        --student-protocol-name imagenette_cvdd_random_real_v1 --adamw-lr-override 5e-4 --adamw-weight-decay 0.01 \
        --adamw-beta1 0.9 --adamw-beta2 0.999 --adamw-eps 1e-8 --cos --eta-override "$eta" \
        --val-dir "$DATA_ROOT/test" --disable-wandb --per-class-output "$result" > "$log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/record_imagenette_random_cvdd_result.py" --result "$result" \
    --ipc "$ipc" --student-seed "$seed" --image-root "$(images "$ipc")" --fkd "$(fkd "$ipc")" \
    --selection-manifest "$(manifest "$ipc")" --teacher "$TEACHER" >> "$log" 2>&1
}

native_one(){
  local ipc="$1" seed="$2" gpu="$3" result log
  result="$EXP_ROOT/results/rded_native/random_real/ipc${ipc}_sseed${seed}/result.json"
  log="$LOG_ROOT/native_ipc${ipc}_sseed${seed}.log"; mkdir -p "$(dirname "$result")"
  exec 7>"${result}.lock"; flock -n 7 || return 75
  if [[ ! -f "$result" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -u "$ROOT_DIR/RDED/validation/train_imagenette_native_full.py" --train-dir "$(images "$ipc")" \
        --val-dir "$DATA_ROOT/test" --teacher "$TEACHER" --output-dir "$(dirname "$result")" \
        --ipc "$ipc" --student-seed "$seed" --image-arm random_real --workers 4 > "$log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/audit_imagenette_rded_native_result.py" --result "$result" \
    --ipc "$ipc" --seed "$seed" --arm random_real >> "$log" 2>&1
}

main(){
  exec 9>"$LOCK_ROOT/random_launcher.lock"; flock -n 9 || exit 75
  rm -f "$STATUS_ROOT/launcher.failed" "$STATUS_ROOT/launcher.complete"
  echo "$(timestamp) started; selection_seed=42; expected_new_results=12" > "$STATUS_ROOT/launcher.running"
  prepare; echo "$(timestamp) complete" > "$STATUS_ROOT/selection.complete"
  local pids=() ipc gpu seed index=0
  for i in 0 1; do ipc="${IPCS[$i]}"; gpu="${GPUS[$i]}"; relabel_one "$ipc" "$gpu" & pids+=("$!"); done
  wait_all "${pids[@]}"; echo "$(timestamp) complete" > "$STATUS_ROOT/relabel.complete"
  pids=()
  for ipc in "${IPCS[@]}"; do for seed in "${STUDENT_SEEDS[@]}"; do
    gpu="${GPUS[$((index%${#GPUS[@]}))]}"; current_one "$ipc" "$seed" "$gpu" & pids+=("$!"); index=$((index+1))
    if (( ${#pids[@]} == ${#GPUS[@]}*3 )); then wait_all "${pids[@]}"; pids=(); fi
  done; done
  for ipc in "${IPCS[@]}"; do for seed in "${STUDENT_SEEDS[@]}"; do
    gpu="${GPUS[$((index%${#GPUS[@]}))]}"; native_one "$ipc" "$seed" "$gpu" & pids+=("$!"); index=$((index+1))
    if (( ${#pids[@]} == ${#GPUS[@]}*3 )); then wait_all "${pids[@]}"; pids=(); fi
  done; done
  if (( ${#pids[@]} )); then wait_all "${pids[@]}"; fi
  python "$ROOT_DIR/CV-DD/fine_grained/summarize_imagenette_random_protocol.py" \
    --root "$EXP_ROOT" --ab-root "$AB_ROOT" > "$LOG_ROOT/summary.log" 2>&1
  rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) complete" > "$STATUS_ROOT/launcher.complete"
}

trap 'rc=$?; if ((rc)); then rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) exit=$rc" > "$STATUS_ROOT/launcher.failed"; fi' EXIT
main
