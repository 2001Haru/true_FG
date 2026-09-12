#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_LGM_soft_v2/cub_ipc3_v1}"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
TEACHER_DIR="${TEACHER_DIR:-/linxi/dataset/FG_SRe2L_standard/v1/teachers/CUB_imsize224/tseed42}"
RANDOM_ROOT="${RANDOM_ROOT:-/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard}"
ARCHIVE="$EXP_ROOT/upload/CUB200_LGM_global_600_images_20260911.zip"
LGM_ROOT="$EXP_ROOT/input"
GPUS=(${GPUS:-0 1}); EVALS_PER_GPU="${EVALS_PER_GPU:-4}"
LOG_ROOT="$EXP_ROOT/logs"; STATUS_ROOT="$EXP_ROOT/status"; LOCK_ROOT="$EXP_ROOT/locks"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$LOCK_ROOT" "$EXP_ROOT/summary"

timestamp(){ date --iso-8601=seconds; }
wait_all(){ local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; ((failed==0)); }
images(){ if [[ "$1" == lgm ]]; then echo "$LGM_ROOT/selected"; else echo "$RANDOM_ROOT/selected/CUB_imsize224/rseed${2}/ipc3"; fi; }
manifest(){ if [[ "$1" == lgm ]]; then echo "$LGM_ROOT/lgm_input_manifest.json"; else echo "$RANDOM_ROOT/manifests/CUB_imsize224/rseed${2}/ipc3.json"; fi; }
fkd(){ local suffix=""; [[ "$2" != none ]] && suffix="_rseed${2}"; echo "$EXP_ROOT/fkd/${1}${suffix}/ipc3_bs20_ipc3"; }

prepare(){
  python "$ROOT_DIR/CV-DD/fine_grained/prepare_lgm_cub_ipc3.py" --archive "$ARCHIVE" \
    --reference-train "$DATA_ROOT/CUB_imsize224/train" --output-root "$LGM_ROOT" > "$LOG_ROOT/input_audit.log" 2>&1
}

relabel_one(){
  local method="$1" source_seed="$2" gpu="$3" suffix="" base actual expected=12000 count=0
  [[ "$source_seed" != none ]] && suffix="_rseed${source_seed}"
  base="$EXP_ROOT/fkd/${method}${suffix}/ipc3"; actual="$(fkd "$method" "$source_seed")"
  [[ -d "$actual" ]] && count="$(find "$actual" -type f -name 'batch_*.tar' | wc -l)"
  if (( count != expected )); then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      python -u "$ROOT_DIR/CV-DD/relabel/relabel.py" --syn-data-path "$(images "$method" "$source_seed")" \
        --fkd-path "$base" --model-pool-dir "$TEACHER_DIR" --teacher-model-name ResNet18 \
        --gpu 0 --batch-size 20 --workers 8 --dataset-name CUB_imsize224 --epochs 400 \
        --seed 42 --fkd-seed 42 --min-scale-crops 0.08 --max-scale-crops 1 \
        --use-fp16 --mode fkd_save --mix-type cutmix > "$LOG_ROOT/relabel_${method}_src${source_seed}.log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$actual" \
    --images 600 --classes 200 --batch-size 20 --epochs 400 >> "$LOG_ROOT/relabel_${method}_src${source_seed}.log" 2>&1
}

eval_one(){
  local method="$1" source_seed="$2" student="$3" gpu="$4" suffix="" result log
  [[ "$source_seed" != none ]] && suffix="/rseed${source_seed}"
  result="$EXP_ROOT/results/${method}${suffix}/ipc3_sseed${student}.json"
  log="$LOG_ROOT/eval_${method}_src${source_seed}_sseed${student}.log"; mkdir -p "$(dirname "$result")"
  exec 7>"${result}.lock"; flock -n 7 || return 75
  if [[ ! -f "$result" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -u "$ROOT_DIR/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 \
        --exp-name "cub_${method}_src${source_seed}_soft_v2_s${student}" --original-data-path "$(images "$method" "$source_seed")" \
        --fkd-path "$(fkd "$method" "$source_seed")" --output-dir "$EXP_ROOT/post_eval" --batch-size 20 --epochs 400 \
        --dataset-name CUB_imsize224 --gradient-accumulation-steps 2 --mix-type cutmix --workers 8 \
        --persistent-workers --fkd_seed 42 --train-seed "$student" --temperature 20 \
        --student-initialization imagenet-v1 --student-protocol-name standard_protocol_v2 \
        --adamw-weight-decay 1e-5 --adamw-beta1 0.9 --adamw-beta2 0.999 --adamw-eps 1e-8 \
        --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 \
        --val-dir "$DATA_ROOT/CUB_imsize224/test" --disable-wandb --per-class-output "$result" > "$log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/record_cub_external_soft_v2_result.py" --result "$result" \
    --source-manifest "$(manifest "$method" "$source_seed")" --fkd "$(fkd "$method" "$source_seed")" \
    --teacher "$TEACHER_DIR/ResNet18.pth" --method "$method" --source-seed "$source_seed" \
    --student-seed "$student" >> "$log" 2>&1
}

main(){
  exec 9>"$LOCK_ROOT/launcher.lock"; flock -n 9 || exit 75
  rm -f "$STATUS_ROOT/launcher.failed" "$STATUS_ROOT/launcher.complete"
  echo "$(timestamp) started" > "$STATUS_ROOT/launcher.running"
  prepare; echo "$(timestamp) complete" > "$STATUS_ROOT/input_audit.complete"
  local pids=() index=0 method source_seed gpu student
  for method in lgm random_real; do
    if [[ "$method" == lgm ]]; then source_seeds=(none); else source_seeds=(0 1 2); fi
    for source_seed in "${source_seeds[@]}"; do
      gpu="${GPUS[$((index%${#GPUS[@]}))]}"; relabel_one "$method" "$source_seed" "$gpu" & pids+=("$!"); index=$((index+1))
      if (( ${#pids[@]} == ${#GPUS[@]} )); then wait_all "${pids[@]}"; pids=(); fi
    done
  done
  if (( ${#pids[@]} )); then wait_all "${pids[@]}"; fi
  echo "$(timestamp) complete" > "$STATUS_ROOT/relabel.complete"
  pids=(); index=0
  for method in lgm random_real; do
    if [[ "$method" == lgm ]]; then source_seeds=(none); else source_seeds=(0 1 2); fi
    for source_seed in "${source_seeds[@]}"; do for student in 42 43 44; do
      gpu="${GPUS[$((index%${#GPUS[@]}))]}"; eval_one "$method" "$source_seed" "$student" "$gpu" & pids+=("$!"); index=$((index+1))
      if (( ${#pids[@]} == ${#GPUS[@]}*EVALS_PER_GPU )); then wait_all "${pids[@]}"; pids=(); fi
    done; done
  done
  if (( ${#pids[@]} )); then wait_all "${pids[@]}"; fi
  python "$ROOT_DIR/CV-DD/fine_grained/summarize_cub_lgm_random_soft_v2.py" --root "$EXP_ROOT" > "$LOG_ROOT/summary.log" 2>&1
  rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) complete" > "$STATUS_ROOT/launcher.complete"
}

trap 'rc=$?; if ((rc)); then rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) exit=$rc" > "$STATUS_ROOT/launcher.failed"; fi' EXIT
main
