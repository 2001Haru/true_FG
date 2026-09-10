#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_RDED_original_v2/aircraft_v1}"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
TEACHER_DIR="${TEACHER_DIR:-/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42}"
EVALS_PER_GPU="${EVALS_PER_GPU:-4}"
GPUS=(${GPUS:-0 1})
GENERATION_SEEDS=(42 43 44)
STUDENT_SEEDS=(42 43 44)
IPCS=(1 3 5)
LOG_ROOT="$EXP_ROOT/logs"
STATUS_ROOT="$EXP_ROOT/status"
LOCK_ROOT="$EXP_ROOT/locks"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$LOCK_ROOT" "$EXP_ROOT/summary"

timestamp(){ date --iso-8601=seconds; }
wait_all(){ local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; ((failed==0)); }

write_definition(){
python - "$EXP_ROOT" "$DATA_ROOT" "$TEACHER_DIR" "$(git -C "$ROOT_DIR" rev-parse HEAD)" <<'PY'
import json, os, sys
from pathlib import Path
root,data,teacher=map(Path,sys.argv[1:4]); revision=sys.argv[4]
expected=[str((root/'results/A_imsize224'/f'gseed{g}'/f'ipc{i}_sseed{s}.json').resolve())
          for i in (1,3,5) for g in (42,43,44) for s in (42,43,44)]
payload={
  'status':'running','experiment':'aircraft_released_rded_generation_soft_v2',
  'git_revision':revision,'dataset':'A_imsize224','teacher_seed':42,
  'generation_seeds':[42,43,44],'student_seeds':[42,43,44],'ipcs':[1,3,5],
  'generation':{'method':'released_rded_joint_topk_mosaic','factor':2,'num_crops':5,
                'candidate_count_per_class':66,'pixel_optimization':False,'bn_matching':False},
  'data_root':str(data.resolve()),'teacher_dir':str(teacher.resolve()),
  'supervision':'SRe2L++ Teacher42 BSSL/FKD soft labels','student_protocol':'v2',
  'expected_generation_manifests':3,'expected_fkd_sets':9,'expected_results':27,
  'expected_result_files':expected,
  'summary':str((root/'summary/rded_original_v2.json').resolve())}
p=root/'matrix_definition.json'; t=p.with_suffix('.json.tmp')
t.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
}

generate_one(){
  local seed="$1" gpu="$2" output="$EXP_ROOT/generated/A_imsize224/gseed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    python -u "$ROOT_DIR/RDED/synthesize/aircraft_original.py" \
      --data-dir "$DATA_ROOT/A_imsize224" --teacher "$TEACHER_DIR/ResNet18.pth" \
      --output-root "$output" --generation-seed "$seed" --forward-batch-size 132 \
      --skip-completed > "$LOG_ROOT/generate_gseed${seed}.log" 2>&1
}

relabel_one(){
  local ipc="$1" generation="$2" gpu="$3"
  local selected="$EXP_ROOT/generated/A_imsize224/gseed${generation}/ipc${ipc}"
  local manifest="$EXP_ROOT/generated/A_imsize224/gseed${generation}/generation_manifest.json"
  local base="$EXP_ROOT/fkd/A_imsize224/gseed${generation}/ipc${ipc}"
  local actual="${base}_bs20_ipc${ipc}" expected=$((400*100*ipc/20)) count=0
  [[ -f "$manifest" && -d "$selected" && -f "$TEACHER_DIR/ResNet18.pth" ]]
  [[ -d "$actual" ]] && count="$(find "$actual" -type f -name 'batch_*.tar' | wc -l)"
  if (( count != expected )); then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      python -u "$ROOT_DIR/CV-DD/relabel/relabel.py" \
        --syn-data-path "$selected" --fkd-path "$base" --model-pool-dir "$TEACHER_DIR" \
        --teacher-model-name ResNet18 --gpu 0 --batch-size 20 --workers 8 \
        --dataset-name A_imsize224 --epochs 400 --seed 42 --fkd-seed 42 \
        --min-scale-crops 0.08 --max-scale-crops 1 --use-fp16 --mode fkd_save \
        --mix-type cutmix > "$LOG_ROOT/relabel_ipc${ipc}_gseed${generation}.log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$actual" \
    --images $((100*ipc)) --classes 100 --batch-size 20 --epochs 400 \
    >> "$LOG_ROOT/relabel_ipc${ipc}_gseed${generation}.log" 2>&1
}

eval_one(){
  local ipc="$1" generation="$2" student="$3" gpu="$4"
  local selected="$EXP_ROOT/generated/A_imsize224/gseed${generation}/ipc${ipc}"
  local manifest="$EXP_ROOT/generated/A_imsize224/gseed${generation}/generation_manifest.json"
  local fkd="$EXP_ROOT/fkd/A_imsize224/gseed${generation}/ipc${ipc}_bs20_ipc${ipc}"
  local result="$EXP_ROOT/results/A_imsize224/gseed${generation}/ipc${ipc}_sseed${student}.json"
  local log="$LOG_ROOT/eval_ipc${ipc}_gseed${generation}_sseed${student}.log"
  mkdir -p "$(dirname "$result")"
  exec 7>"${result}.lock"; flock -n 7 || return 75
  if [[ ! -f "$result" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -u "$ROOT_DIR/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc "$ipc" \
        --exp-name "rded_original_v2_A_ipc${ipc}_g${generation}_s${student}" \
        --original-data-path "$selected" --fkd-path "$fkd" --output-dir "$EXP_ROOT/post_eval" \
        --batch-size 20 --epochs 400 --dataset-name A_imsize224 \
        --gradient-accumulation-steps 2 --mix-type cutmix --workers 8 --persistent-workers \
        --fkd_seed 42 --train-seed "$student" --temperature 20 \
        --student-initialization imagenet-v1 --student-protocol-name standard_protocol_v2 \
        --adamw-weight-decay 1e-5 --adamw-beta1 0.9 --adamw-beta2 0.999 --adamw-eps 1e-8 \
        --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 \
        --cosine-t-max 400 --cosine-eta-min 0 \
        --val-dir "$DATA_ROOT/A_imsize224/test" --disable-wandb \
        --per-class-output "$result" > "$log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/record_rded_original_v2_result.py" \
    --result "$result" --generation-manifest "$manifest" --teacher-dir "$TEACHER_DIR" \
    --fkd-dir "$fkd" --ipc "$ipc" --generation-seed "$generation" \
    --student-seed "$student" >> "$log" 2>&1
}

main(){
  exec 9>"$LOCK_ROOT/launcher.lock"; flock -n 9 || { echo 'launcher already running' >&2; exit 75; }
  write_definition
  [[ -d "$DATA_ROOT/A_imsize224/train" && -d "$DATA_ROOT/A_imsize224/test" ]]
  [[ -f "$TEACHER_DIR/ResNet18.pth" && -f "$TEACHER_DIR/complete.json" ]]
  echo "$(timestamp) started" > "$STATUS_ROOT/launcher.running"
  rm -f "$STATUS_ROOT/launcher.failed" "$STATUS_ROOT/launcher.complete"

  local pids=() index=0 seed ipc generation student gpu
  for seed in "${GENERATION_SEEDS[@]}"; do
    gpu="${GPUS[$((index%${#GPUS[@]}))]}"; generate_one "$seed" "$gpu" & pids+=("$!"); index=$((index+1))
    if (( ${#pids[@]} == ${#GPUS[@]} )); then wait_all "${pids[@]}"; pids=(); fi
  done
  if (( ${#pids[@]} )); then wait_all "${pids[@]}"; fi
  echo "$(timestamp) generation complete" > "$STATUS_ROOT/generation.complete"

  pids=(); index=0
  for ipc in "${IPCS[@]}"; do for generation in "${GENERATION_SEEDS[@]}"; do
    gpu="${GPUS[$((index%${#GPUS[@]}))]}"; relabel_one "$ipc" "$generation" "$gpu" & pids+=("$!"); index=$((index+1))
    if (( ${#pids[@]} == ${#GPUS[@]} )); then wait_all "${pids[@]}"; pids=(); fi
  done; done
  if (( ${#pids[@]} )); then wait_all "${pids[@]}"; fi
  echo "$(timestamp) relabel complete" > "$STATUS_ROOT/relabel.complete"

  pids=(); index=0
  for ipc in "${IPCS[@]}"; do for generation in "${GENERATION_SEEDS[@]}"; do for student in "${STUDENT_SEEDS[@]}"; do
    gpu="${GPUS[$((index%${#GPUS[@]}))]}"; eval_one "$ipc" "$generation" "$student" "$gpu" & pids+=("$!"); index=$((index+1))
    if (( ${#pids[@]} == ${#GPUS[@]}*EVALS_PER_GPU )); then wait_all "${pids[@]}"; pids=(); fi
  done; done; done
  if (( ${#pids[@]} )); then wait_all "${pids[@]}"; fi
  python "$ROOT_DIR/CV-DD/fine_grained/summarize_rded_original_v2.py" --root "$EXP_ROOT" \
    > "$LOG_ROOT/summary.log" 2>&1
  python - "$EXP_ROOT/matrix_definition.json" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads(p.read_text()); x['status']='complete'
t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
  rm -f "$STATUS_ROOT/launcher.running"
  echo "$(timestamp) complete" > "$STATUS_ROOT/launcher.complete"
}

trap 'rc=$?; if ((rc)); then rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) exit=$rc" > "$STATUS_ROOT/launcher.failed"; fi' EXIT
main
