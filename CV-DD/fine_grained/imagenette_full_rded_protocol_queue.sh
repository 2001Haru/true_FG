#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_ImageNette_full_RDED_protocol/v1}"
AB_ROOT="${AB_ROOT:-/linxi/dataset/FG_ImageNette_native_resolution/v1}"
C_ROOT="${C_ROOT:-/linxi/dataset/FG_ImageNette_same_source_mosaic/v1}"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/CV-DD/test_data/imagenet-nette}"
TEACHER="${TEACHER:-/linxi/dataset/CV-DD/offline_models/imagenet-nette/ResNet18.pth}"
GEN_ROOT="$EXP_ROOT/generation"
GPUS=(${GPUS:-0 1})
IPCS=(10 50); STUDENT_SEEDS=(42 43 44)
LOG_ROOT="$EXP_ROOT/logs"; STATUS_ROOT="$EXP_ROOT/status"; LOCK_ROOT="$EXP_ROOT/locks"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$LOCK_ROOT" "$EXP_ROOT/summary"

timestamp(){ date --iso-8601=seconds; }
wait_all(){ local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; ((failed==0)); }
d_images(){ echo "$GEN_ROOT/ipc$1"; }
d_fkd(){ echo "$EXP_ROOT/fkd/current_cvdd/d_rded/ipc${1}_bs10_ipc${1}"; }
a_images(){ echo "$AB_ROOT/construction/selected/a_image/ipc$1"; }

write_definition(){
python - "$EXP_ROOT" "$AB_ROOT" "$C_ROOT" "$DATA_ROOT" "$TEACHER" "$(git -C "$ROOT_DIR" rev-parse HEAD)" <<'PY'
import json,os,sys
from pathlib import Path
root,ab,c,data,teacher=map(Path,sys.argv[1:6]); revision=sys.argv[6]
x={'status':'running','experiment':'imagenette_full_rded_x_downstream_protocol','git_revision':revision,
   'dataset':'imagenet-nette','generation_seed':42,'ipcs':[10,50],'student_seeds':[42,43,44],
   'generation':{'candidates_per_class':300,'crops_per_candidate':5,'best_crop_score':'true-class CE',
     'scoring_input':'ImageNet-normalized 112 crop center-zero-padded to 224','source_pre_crop_resize':'none',
     'joint_topk':{'10':40,'50':200},'factor':2},
   'new_conditions':['D/current_cvdd','A/rded_native','D/rded_native'],'new_student_trainings':18,
   'current_cvdd':{'batch':10,'accumulation':2,'lr':5e-4,'eta':{'10':1,'50':2},'teacher':'train BSSL offline FKD','rrc':[.08,1]},
   'rded_native':{'batch':{'10':50,'50':100},'accumulation':1,'lr':1e-3,'eta':2,
     'teacher':'eval online','rrc':[.5,1],'shuffle_patches':'D only','native_extra_student_bn_forward_on_mix':True},
   'validation_images':3925,'ab_root':str(ab.resolve()),'c_root':str(c.resolve()),
   'data_root':str(data.resolve()),'teacher':str(teacher.resolve()),
   'summary':str((root/'summary/imagenette_full_rded_protocol.json').resolve())}
p=root/'experiment_definition.json'; t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
}

generate(){
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -u "$ROOT_DIR/RDED/synthesize/imagenette_original.py" \
      --data-root "$DATA_ROOT" --teacher "$TEACHER" --output-root "$GEN_ROOT" \
      --generation-seed 42 --forward-batch-size 300 --skip-completed > "$LOG_ROOT/generation.log" 2>&1
}

relabel_one(){
  local ipc="$1" gpu="$2" base="$EXP_ROOT/fkd/current_cvdd/d_rded/ipc${ipc}" actual expected count=0
  actual="$(d_fkd "$ipc")"; expected=$((300*ipc))
  [[ -d "$actual" ]] && count="$(find "$actual" -type f -name 'batch_*.tar' | wc -l)"
  if (( count != expected )); then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      python -u "$ROOT_DIR/CV-DD/relabel/relabel.py" --syn-data-path "$(d_images "$ipc")" \
        --fkd-path "$base" --model-pool-dir "$(dirname "$TEACHER")" --teacher-model-name ResNet18 \
        --gpu 0 --batch-size 10 --workers 2 --dataset-name imagenet-nette --epochs 300 \
        --seed 42 --fkd-seed 42 --min-scale-crops 0.08 --max-scale-crops 1 \
        --use-fp16 --mode fkd_save --mix-type cutmix > "$LOG_ROOT/relabel_current_ipc${ipc}.log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$actual" \
    --images $((10*ipc)) --classes 10 --batch-size 10 --epochs 300 >> "$LOG_ROOT/relabel_current_ipc${ipc}.log" 2>&1
}

current_one(){
  local ipc="$1" seed="$2" gpu="$3" eta result log
  [[ "$ipc" == 10 ]] && eta=1 || eta=2
  result="$EXP_ROOT/results/current_cvdd/d_rded/ipc${ipc}_sseed${seed}.json"
  log="$LOG_ROOT/current_d_ipc${ipc}_sseed${seed}.log"; mkdir -p "$(dirname "$result")"
  exec 7>"${result}.lock"; flock -n 7 || return 75
  if [[ ! -f "$result" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -u "$ROOT_DIR/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc "$ipc" \
        --exp-name "imagenette_full_rded_current_ipc${ipc}_s${seed}" \
        --original-data-path "$(d_images "$ipc")" --fkd-path "$(d_fkd "$ipc")" --output-dir "$EXP_ROOT/post_eval" \
        --batch-size 10 --epochs 300 --dataset-name imagenet-nette --gradient-accumulation-steps 2 \
        --mix-type cutmix --workers 2 --fkd_seed 42 --train-seed "$seed" --temperature 20 \
        --student-initialization random --student-protocol-name imagenette_cvdd_full_rded_v1 \
        --adamw-lr-override 5e-4 --adamw-weight-decay 0.01 --adamw-beta1 0.9 --adamw-beta2 0.999 --adamw-eps 1e-8 \
        --cos --eta-override "$eta" --val-dir "$DATA_ROOT/test" --disable-wandb --per-class-output "$result" > "$log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/record_imagenette_full_rded_cvdd.py" --result "$result" \
    --ipc "$ipc" --student-seed "$seed" --image-root "$(d_images "$ipc")" --fkd "$(d_fkd "$ipc")" \
    --generation-manifest "$GEN_ROOT/generation_manifest.json" --teacher "$TEACHER" >> "$log" 2>&1
}

native_one(){
  local ipc="$1" arm="$2" seed="$3" gpu="$4" images result log
  [[ "$arm" == a_original ]] && images="$(a_images "$ipc")" || images="$(d_images "$ipc")"
  result="$EXP_ROOT/results/rded_native/$arm/ipc${ipc}_sseed${seed}/result.json"
  log="$LOG_ROOT/native_${arm}_ipc${ipc}_sseed${seed}.log"; mkdir -p "$(dirname "$result")"
  exec 7>"${result}.lock"; flock -n 7 || return 75
  if [[ ! -f "$result" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -u "$ROOT_DIR/RDED/validation/train_imagenette_native_full.py" --train-dir "$images" \
        --val-dir "$DATA_ROOT/test" --teacher "$TEACHER" --output-dir "$(dirname "$result")" \
        --ipc "$ipc" --student-seed "$seed" --image-arm "$arm" --workers 4 > "$log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/audit_imagenette_rded_native_result.py" --result "$result" \
    --ipc "$ipc" --seed "$seed" --arm "$arm" >> "$log" 2>&1
}

main(){
  exec 9>"$LOCK_ROOT/launcher.lock"; flock -n 9 || exit 75
  write_definition; rm -f "$STATUS_ROOT/launcher.failed" "$STATUS_ROOT/launcher.complete"
  echo "$(timestamp) started" > "$STATUS_ROOT/launcher.running"
  generate; echo "$(timestamp) complete" > "$STATUS_ROOT/generation.complete"
  local pids=() ipc gpu
  for i in 0 1; do ipc="${IPCS[$i]}"; gpu="${GPUS[$i]}"; relabel_one "$ipc" "$gpu" & pids+=("$!"); done
  wait_all "${pids[@]}"; echo "$(timestamp) complete" > "$STATUS_ROOT/relabel.complete"
  pids=(); local index=0 seed arm
  for ipc in "${IPCS[@]}"; do for seed in "${STUDENT_SEEDS[@]}"; do
    gpu="${GPUS[$((index%${#GPUS[@]}))]}"; current_one "$ipc" "$seed" "$gpu" & pids+=("$!"); index=$((index+1))
    if (( ${#pids[@]} == ${#GPUS[@]}*3 )); then wait_all "${pids[@]}"; pids=(); fi
  done; done
  for ipc in "${IPCS[@]}"; do for arm in a_original d_rded; do for seed in "${STUDENT_SEEDS[@]}"; do
    gpu="${GPUS[$((index%${#GPUS[@]}))]}"; native_one "$ipc" "$arm" "$seed" "$gpu" & pids+=("$!"); index=$((index+1))
    if (( ${#pids[@]} == ${#GPUS[@]}*3 )); then wait_all "${pids[@]}"; pids=(); fi
  done; done; done
  if (( ${#pids[@]} )); then wait_all "${pids[@]}"; fi
  python "$ROOT_DIR/CV-DD/fine_grained/summarize_imagenette_full_rded_protocol.py" \
    --root "$EXP_ROOT" --ab-root "$AB_ROOT" --c-root "$C_ROOT" > "$LOG_ROOT/summary.log" 2>&1
  python - "$EXP_ROOT/experiment_definition.json" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads(p.read_text()); x['status']='complete'; t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
  rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) complete" > "$STATUS_ROOT/launcher.complete"
}

trap 'rc=$?; if ((rc)); then rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) exit=$rc" > "$STATUS_ROOT/launcher.failed"; fi' EXIT
main
