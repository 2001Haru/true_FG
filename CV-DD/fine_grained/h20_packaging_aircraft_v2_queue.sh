#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_H20_packaging_v2/aircraft_ipc5_v1}"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
TEACHER_DIR="${TEACHER_DIR:-/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42}"
H20_ROOT="${H20_ROOT:-/linxi/dataset/FG_TeacherScore_selection_v2/aircraft_v1}"
SOURCE_IMAGES="$H20_ROOT/selection/selected/entropy_t20/ipc5"
SOURCE_MANIFEST="$H20_ROOT/selection/selection_manifest.json"
CONSTRUCTION_ROOT="$EXP_ROOT/construction"
CONSTRUCTION_MANIFEST="$CONSTRUCTION_ROOT/construction_manifest.json"
GPUS=(${GPUS:-0 1})
ARMS=(downsample_up same_source_mosaic)
STUDENT_SEEDS=(42 43 44)
LOG_ROOT="$EXP_ROOT/logs"; STATUS_ROOT="$EXP_ROOT/status"; LOCK_ROOT="$EXP_ROOT/locks"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$LOCK_ROOT" "$EXP_ROOT/summary"

timestamp(){ date --iso-8601=seconds; }
wait_all(){ local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; ((failed==0)); }

write_definition(){
python - "$EXP_ROOT" "$SOURCE_IMAGES" "$SOURCE_MANIFEST" "$TEACHER_DIR" "$(git -C "$ROOT_DIR" rev-parse HEAD)" <<'PY'
import json,os,sys
from pathlib import Path
root,source,manifest,teacher=map(Path,sys.argv[1:5]); revision=sys.argv[5]
x={'status':'running','experiment':'aircraft_h20_ipc5_packaging_soft_v2','git_revision':revision,
   'dataset':'A_imsize224','ipc':5,'teacher_seed':42,'student_seeds':[42,43,44],
   'source_selection':'entropy_t20','source_images':str(source.resolve()),
   'source_manifest':str(manifest.resolve()),'teacher_dir':str(teacher.resolve()),
   'arms':{'A':'existing original 224x224 H20 IPC5','B':'same images 224->112->224 bilinear',
           'C':'same 112 cache, five balanced leave-one-out 2x2 mosaics'},
   'rrc_scale':[0.08,1.0],'new_fkd_sets':2,'new_student_trainings':6,
   'expected_result_files':[str((root/f'results/{arm}/ipc5_sseed{s}.json').resolve())
                            for arm in ('downsample_up','same_source_mosaic') for s in (42,43,44)],
   'summary':str((root/'summary/h20_packaging_v2.json').resolve())}
p=root/'experiment_definition.json'; t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
}

relabel_one(){
  local arm="$1" gpu="$2"
  local images="$CONSTRUCTION_ROOT/$arm"
  local base="$EXP_ROOT/fkd/$arm/ipc5"
  local actual="${base}_bs20_ipc5" expected=10000 count=0 log="$LOG_ROOT/relabel_${arm}.log"
  [[ -d "$actual" ]] && count="$(find "$actual" -type f -name 'batch_*.tar' | wc -l)"
  if (( count != expected )); then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      python -u "$ROOT_DIR/CV-DD/relabel/relabel.py" --syn-data-path "$images" \
        --fkd-path "$base" --model-pool-dir "$TEACHER_DIR" --teacher-model-name ResNet18 \
        --gpu 0 --batch-size 20 --workers 8 --dataset-name A_imsize224 --epochs 400 \
        --seed 42 --fkd-seed 42 --min-scale-crops 0.08 --max-scale-crops 1 \
        --use-fp16 --mode fkd_save --mix-type cutmix > "$log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$actual" \
    --images 500 --classes 100 --batch-size 20 --epochs 400 >> "$log" 2>&1
}

eval_one(){
  local arm="$1" student="$2" gpu="$3"
  local images="$CONSTRUCTION_ROOT/$arm"
  local fkd="$EXP_ROOT/fkd/$arm/ipc5_bs20_ipc5"
  local result="$EXP_ROOT/results/$arm/ipc5_sseed${student}.json"
  local log="$LOG_ROOT/eval_${arm}_sseed${student}.log"
  mkdir -p "$(dirname "$result")"; exec 7>"${result}.lock"; flock -n 7 || return 75
  if [[ ! -f "$result" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -u "$ROOT_DIR/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 5 \
        --exp-name "h20_${arm}_A_ipc5_s${student}" --original-data-path "$images" \
        --fkd-path "$fkd" --output-dir "$EXP_ROOT/post_eval" --batch-size 20 --epochs 400 \
        --dataset-name A_imsize224 --gradient-accumulation-steps 2 --mix-type cutmix \
        --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$student" --temperature 20 \
        --student-initialization imagenet-v1 --student-protocol-name standard_protocol_v2 \
        --adamw-weight-decay 1e-5 --adamw-beta1 0.9 --adamw-beta2 0.999 --adamw-eps 1e-8 \
        --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 \
        --val-dir "$DATA_ROOT/A_imsize224/test" --disable-wandb --per-class-output "$result" \
        > "$log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/record_h20_packaging_v2_result.py" \
    --result "$result" --arm "$arm" --construction-manifest "$CONSTRUCTION_MANIFEST" \
    --teacher-dir "$TEACHER_DIR" --fkd-dir "$fkd" --student-seed "$student" >> "$log" 2>&1
}

main(){
  exec 9>"$LOCK_ROOT/launcher.lock"; flock -n 9 || exit 75
  write_definition; rm -f "$STATUS_ROOT/launcher.failed" "$STATUS_ROOT/launcher.complete"
  echo "$(timestamp) started" > "$STATUS_ROOT/launcher.running"
  python -u "$ROOT_DIR/CV-DD/fine_grained/prepare_h20_ipc5_packaging.py" \
    --source-root "$SOURCE_IMAGES" --source-manifest "$SOURCE_MANIFEST" \
    --output-root "$CONSTRUCTION_ROOT" --skip-completed > "$LOG_ROOT/construction.log" 2>&1
  echo "$(timestamp) construction complete" > "$STATUS_ROOT/construction.complete"
  local pids=() index=0 arm student gpu
  for arm in "${ARMS[@]}"; do
    gpu="${GPUS[$index]}"; relabel_one "$arm" "$gpu" & pids+=("$!"); index=$((index+1))
  done
  wait_all "${pids[@]}"; echo "$(timestamp) relabel complete" > "$STATUS_ROOT/relabel.complete"
  pids=(); index=0
  for arm in "${ARMS[@]}"; do for student in "${STUDENT_SEEDS[@]}"; do
    gpu="${GPUS[$((index%${#GPUS[@]}))]}"; eval_one "$arm" "$student" "$gpu" & pids+=("$!"); index=$((index+1))
  done; done
  wait_all "${pids[@]}"
  python "$ROOT_DIR/CV-DD/fine_grained/summarize_h20_packaging_v2.py" --root "$EXP_ROOT" \
    --original-root "$H20_ROOT" > "$LOG_ROOT/summary.log" 2>&1
  python - "$EXP_ROOT/experiment_definition.json" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads(p.read_text()); x['status']='complete'; t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
  rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) complete" > "$STATUS_ROOT/launcher.complete"
}

trap 'rc=$?; if ((rc)); then rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) exit=$rc" > "$STATUS_ROOT/launcher.failed"; fi' EXIT
main
