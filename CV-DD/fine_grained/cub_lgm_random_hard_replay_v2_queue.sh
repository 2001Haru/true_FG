#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOFT_ROOT="${SOFT_ROOT:-/linxi/dataset/FG_LGM_soft_v2/cub_ipc3_v1}"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_LGM_hard_replay_v2/cub_ipc3_v1}"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
GPUS=(${GPUS:-0 1}); EVALS_PER_GPU="${EVALS_PER_GPU:-4}"
LOG_ROOT="$EXP_ROOT/logs"; STATUS_ROOT="$EXP_ROOT/status"; LOCK_ROOT="$EXP_ROOT/locks"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$LOCK_ROOT" "$EXP_ROOT/summary"

timestamp(){ date --iso-8601=seconds; }
wait_all(){ local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; ((failed==0)); }
soft_result(){ if [[ "$1" == lgm ]]; then echo "$SOFT_ROOT/results/lgm/ipc3_sseed${3}.json"; else echo "$SOFT_ROOT/results/random_real/rseed${2}/ipc3_sseed${3}.json"; fi; }

eval_one(){
  local method="$1" source_seed="$2" student="$3" gpu="$4" source suffix="" result log images fkd
  source="$(soft_result "$method" "$source_seed" "$student")"
  [[ "$source_seed" != none ]] && suffix="/rseed${source_seed}"
  result="$EXP_ROOT/results/${method}${suffix}/ipc3_sseed${student}.json"
  log="$LOG_ROOT/${method}_src${source_seed}_sseed${student}.log"; mkdir -p "$(dirname "$result")"
  exec 7>"${result}.lock"; flock -n 7 || return 75
  IFS=$'\t' read -r images fkd < <(python - "$source" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(x['synthetic_data_path']+'\t'+x['fkd_path'])
PY
  )
  [[ -d "$images" && -d "$fkd" ]]
  if [[ ! -f "$result" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -u "$ROOT_DIR/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 \
        --exp-name "cub_${method}_src${source_seed}_hard_replay_v2_s${student}" \
        --original-data-path "$images" --fkd-path "$fkd" --output-dir "$EXP_ROOT/post_eval" \
        --batch-size 20 --epochs 400 --dataset-name CUB_imsize224 --gradient-accumulation-steps 2 \
        --mix-type cutmix --fkd-hard-label --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$student" \
        --temperature 20 --student-initialization imagenet-v1 --student-protocol-name standard_protocol_v2_hard_replay \
        --adamw-weight-decay 1e-5 --adamw-beta1 0.9 --adamw-beta2 0.999 --adamw-eps 1e-8 \
        --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 \
        --val-dir "$DATA_ROOT/CUB_imsize224/test" --disable-wandb --per-class-output "$result" > "$log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/record_hard_replay_v2_result.py" --result "$result" \
    --source-soft-result "$source" --method "$method" --source-seed "$source_seed" --ipc 3 \
    --student-seed "$student" --classes 200 --validation-images 5794 >> "$log" 2>&1
}

main(){
  exec 9>"$LOCK_ROOT/launcher.lock"; flock -n 9 || exit 75
  rm -f "$STATUS_ROOT/launcher.failed" "$STATUS_ROOT/launcher.complete"
  echo "$(timestamp) started; expected=12; reused_fkd=4" > "$STATUS_ROOT/launcher.running"
  local pids=() index=0 method source_seed student gpu
  for method in lgm random_real; do
    if [[ "$method" == lgm ]]; then source_seeds=(none); else source_seeds=(0 1 2); fi
    for source_seed in "${source_seeds[@]}"; do for student in 42 43 44; do
      gpu="${GPUS[$((index%${#GPUS[@]}))]}"; eval_one "$method" "$source_seed" "$student" "$gpu" & pids+=("$!"); index=$((index+1))
      if (( ${#pids[@]} == ${#GPUS[@]}*EVALS_PER_GPU )); then wait_all "${pids[@]}"; pids=(); fi
    done; done
  done
  if (( ${#pids[@]} )); then wait_all "${pids[@]}"; fi
  python "$ROOT_DIR/CV-DD/fine_grained/summarize_cub_lgm_random_hard_replay_v2.py" --root "$EXP_ROOT" > "$LOG_ROOT/summary.log" 2>&1
  rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) complete" > "$STATUS_ROOT/launcher.complete"
}

trap 'rc=$?; if ((rc)); then rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) exit=$rc" > "$STATUS_ROOT/launcher.failed"; fi' EXIT
main
