#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_RDED_single_quadrant_v2/aircraft_ipc3_g42_v1}"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
TEACHER_DIR="${TEACHER_DIR:-/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42}"
RDED_ROOT="${RDED_ROOT:-/linxi/dataset/FG_RDED_original_v2/aircraft_v1}"
WHOLE_MOSAIC_ROOT="${WHOLE_MOSAIC_ROOT:-/linxi/dataset/FG_RRC_sensitivity_v2/aircraft_ipc3_v1}"
GPUS=(${GPUS:-0 1})
STUDENT_SEEDS=(42 43 44)
IMAGES="$RDED_ROOT/generated/A_imsize224/gseed42/ipc3"
GENERATION_MANIFEST="$RDED_ROOT/generated/A_imsize224/gseed42/generation_manifest.json"
FKD_BASE="$EXP_ROOT/fkd/ipc3_single_quadrant_rrc05"
FKD_DIR="${FKD_BASE}_bs20_ipc3"
LOG_ROOT="$EXP_ROOT/logs"; STATUS_ROOT="$EXP_ROOT/status"; LOCK_ROOT="$EXP_ROOT/locks"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$LOCK_ROOT" "$EXP_ROOT/summary"

timestamp(){ date --iso-8601=seconds; }
wait_all(){ local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; ((failed==0)); }

write_definition(){
python - "$EXP_ROOT" "$IMAGES" "$GENERATION_MANIFEST" "$TEACHER_DIR" "$(git -C "$ROOT_DIR" rev-parse HEAD)" <<'PY'
import json,os,sys
from pathlib import Path
root,images,manifest,teacher=map(Path,sys.argv[1:5]); revision=sys.argv[5]
x={'status':'running','experiment':'aircraft_rded_single_quadrant_soft_v2','git_revision':revision,
   'dataset':'A_imsize224','ipc':3,'teacher_seed':42,'generation_seed':42,
   'student_seeds':[42,43,44],'source_images':str(images.resolve()),
   'generation_manifest':str(manifest.resolve()),'teacher_dir':str(teacher.resolve()),
   'view':{'single_quadrant':True,'quadrant_size':[112,112],'quadrants_expanded':False,
           'samples_per_epoch':300,'balanced_per_epoch':[75,75,75,75],
           'balanced_per_image_over_400_epochs':[100,100,100,100],
           'rrc_scale':[0.5,1.0],'rrc_output_size':[224,224]},
   'new_fkd_sets':1,'new_student_trainings':3,'student_protocol':'standard_protocol_v2',
   'expected_result_files':[str((root/f'results/ipc3_sseed{s}.json').resolve()) for s in (42,43,44)],
   'summary':str((root/'summary/single_quadrant_v2.json').resolve())}
p=root/'experiment_definition.json'; t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
}

relabel(){
  local count=0 expected=6000
  [[ -d "$FKD_DIR" ]] && count="$(find "$FKD_DIR" -type f -name 'batch_*.tar' | wc -l)"
  if (( count != expected )); then
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      python -u "$ROOT_DIR/CV-DD/relabel/relabel.py" --syn-data-path "$IMAGES" \
        --fkd-path "$FKD_BASE" --model-pool-dir "$TEACHER_DIR" --teacher-model-name ResNet18 \
        --gpu 0 --batch-size 20 --workers 8 --dataset-name A_imsize224 --epochs 400 \
        --seed 42 --fkd-seed 42 --single-quadrant --quadrant-seed 42 \
        --min-scale-crops 0.5 --max-scale-crops 1 --use-fp16 --mode fkd_save \
        --mix-type cutmix > "$LOG_ROOT/relabel.log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$FKD_DIR" \
    --images 300 --classes 100 --batch-size 20 --epochs 400 >> "$LOG_ROOT/relabel.log" 2>&1
  python "$ROOT_DIR/CV-DD/fine_grained/audit_quadrant_fkd.py" --fkd-dir "$FKD_DIR" \
    --image-root "$IMAGES" --epochs 400 --batch-size 20 --seed 42 \
    --output "$FKD_DIR/quadrant_audit.json" >> "$LOG_ROOT/relabel.log" 2>&1
}

eval_one(){
  local student="$1" gpu="$2" result="$EXP_ROOT/results/ipc3_sseed${student}.json"
  local log="$LOG_ROOT/eval_sseed${student}.log"
  mkdir -p "$(dirname "$result")"; exec 7>"${result}.lock"; flock -n 7 || return 75
  if [[ ! -f "$result" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -u "$ROOT_DIR/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 \
        --exp-name "rded_single_quadrant_A_ipc3_g42_s${student}" \
        --original-data-path "$IMAGES" --fkd-path "$FKD_DIR" --output-dir "$EXP_ROOT/post_eval" \
        --batch-size 20 --epochs 400 --dataset-name A_imsize224 \
        --gradient-accumulation-steps 2 --mix-type cutmix --workers 8 --persistent-workers \
        --fkd_seed 42 --train-seed "$student" --temperature 20 \
        --student-initialization imagenet-v1 --student-protocol-name standard_protocol_v2 \
        --adamw-weight-decay 1e-5 --adamw-beta1 0.9 --adamw-beta2 0.999 --adamw-eps 1e-8 \
        --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 \
        --val-dir "$DATA_ROOT/A_imsize224/test" --disable-wandb --per-class-output "$result" \
        > "$log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/record_single_quadrant_v2_result.py" \
    --result "$result" --generation-manifest "$GENERATION_MANIFEST" --teacher-dir "$TEACHER_DIR" \
    --fkd-dir "$FKD_DIR" --student-seed "$student" >> "$log" 2>&1
}

main(){
  exec 9>"$LOCK_ROOT/launcher.lock"; flock -n 9 || exit 75
  write_definition; [[ -d "$IMAGES" && -f "$GENERATION_MANIFEST" && -f "$TEACHER_DIR/ResNet18.pth" ]]
  rm -f "$STATUS_ROOT/launcher.failed" "$STATUS_ROOT/launcher.complete"
  echo "$(timestamp) started" > "$STATUS_ROOT/launcher.running"
  relabel; echo "$(timestamp) relabel and quadrant audit complete" > "$STATUS_ROOT/relabel.complete"
  local pids=() index=0 student gpu
  for student in "${STUDENT_SEEDS[@]}"; do
    gpu="${GPUS[$((index%${#GPUS[@]}))]}"; eval_one "$student" "$gpu" & pids+=("$!"); index=$((index+1))
  done
  wait_all "${pids[@]}"
  python "$ROOT_DIR/CV-DD/fine_grained/summarize_single_quadrant_v2.py" \
    --root "$EXP_ROOT" --whole-mosaic-root "$WHOLE_MOSAIC_ROOT" > "$LOG_ROOT/summary.log" 2>&1
  python - "$EXP_ROOT/experiment_definition.json" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads(p.read_text()); x['status']='complete'; t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
  rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) complete" > "$STATUS_ROOT/launcher.complete"
}

trap 'rc=$?; if ((rc)); then rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) exit=$rc" > "$STATUS_ROOT/launcher.failed"; fi' EXIT
main
