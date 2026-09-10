#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_H20_cross_view_labels_v2/aircraft_ipc5_Bimage_Alabel_v1}"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
TEACHER_DIR="${TEACHER_DIR:-/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42}"
A_ROOT="${A_ROOT:-/linxi/dataset/FG_TeacherScore_selection_v2/aircraft_v1}"
B_ROOT="${B_ROOT:-/linxi/dataset/FG_H20_packaging_v2/aircraft_ipc5_v1}"
A_IMAGES="$A_ROOT/selection/selected/entropy_t20/ipc5"
B_IMAGES="$B_ROOT/construction/downsample_up"
A_FKD="$A_ROOT/fkd/entropy_t20/ipc5_bs20_ipc5"
B_FKD="$B_ROOT/fkd/downsample_up/ipc5_bs20_ipc5"
CONSTRUCTION_MANIFEST="$B_ROOT/construction/construction_manifest.json"
ALIGNMENT_AUDIT="$EXP_ROOT/preflight/fkd_alignment.json"
GPUS=(${GPUS:-0 1})
STUDENT_SEEDS=(42 43 44)
LOG_ROOT="$EXP_ROOT/logs"; STATUS_ROOT="$EXP_ROOT/status"; LOCK_ROOT="$EXP_ROOT/locks"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$LOCK_ROOT" "$EXP_ROOT/summary" "$EXP_ROOT/preflight"

timestamp(){ date --iso-8601=seconds; }
wait_all(){ local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; ((failed==0)); }

write_definition(){
python - "$EXP_ROOT" "$A_IMAGES" "$B_IMAGES" "$A_FKD" "$TEACHER_DIR" "$(git -C "$ROOT_DIR" rev-parse HEAD)" <<'PY'
import json,os,sys
from pathlib import Path
root,a,b,fkd,teacher=map(Path,sys.argv[1:6]); revision=sys.argv[6]
x={'status':'running','experiment':'aircraft_h20_B_image_A_label_soft_v2','git_revision':revision,
   'dataset':'A_imsize224','ipc':5,'teacher_seed':42,'student_seeds':[42,43,44],
   'student_images':'B: 224->112->224','teacher_label_images':'A: original 224',
   'a_images':str(a.resolve()),'b_images':str(b.resolve()),'a_fkd':str(fkd.resolve()),
   'teacher_dir':str(teacher.resolve()),'rrc_scale':[0.08,1.0],
   'new_fkd_sets':0,'new_student_trainings':3,
   'alignment_requirement':'all 10000 batch metadata records exactly equal',
   'expected_result_files':[str((root/f'results/ipc5_sseed{s}.json').resolve()) for s in (42,43,44)],
   'summary':str((root/'summary/cross_view_label_v2.json').resolve())}
p=root/'experiment_definition.json'; t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
}

alignment_gate(){
  if [[ -f "$ALIGNMENT_AUDIT" ]] && python - "$ALIGNMENT_AUDIT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); raise SystemExit(0 if x.get('status')=='complete' and x.get('total_mismatches')==0 else 1)
PY
  then return 0; fi
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -u \
    "$ROOT_DIR/CV-DD/fine_grained/audit_cross_view_fkd_alignment.py" \
    --a-images "$A_IMAGES" --b-images "$B_IMAGES" --construction-manifest "$CONSTRUCTION_MANIFEST" \
    --a-fkd "$A_FKD" --b-fkd "$B_FKD" --epochs 400 --batches-per-epoch 25 \
    --output "$ALIGNMENT_AUDIT" > "$LOG_ROOT/alignment.log" 2>&1
}

eval_one(){
  local student="$1" gpu="$2"
  local result="$EXP_ROOT/results/ipc5_sseed${student}.json"
  local log="$LOG_ROOT/eval_sseed${student}.log"
  mkdir -p "$(dirname "$result")"; exec 7>"${result}.lock"; flock -n 7 || return 75
  if [[ ! -f "$result" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -u "$ROOT_DIR/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 5 \
        --exp-name "h20_Bimage_Alabel_A_ipc5_s${student}" --original-data-path "$B_IMAGES" \
        --fkd-path "$A_FKD" --output-dir "$EXP_ROOT/post_eval" --batch-size 20 --epochs 400 \
        --dataset-name A_imsize224 --gradient-accumulation-steps 2 --mix-type cutmix \
        --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$student" --temperature 20 \
        --student-initialization imagenet-v1 --student-protocol-name standard_protocol_v2 \
        --adamw-weight-decay 1e-5 --adamw-beta1 0.9 --adamw-beta2 0.999 --adamw-eps 1e-8 \
        --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 \
        --val-dir "$DATA_ROOT/A_imsize224/test" --disable-wandb --per-class-output "$result" \
        > "$log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/record_cross_view_label_v2_result.py" \
    --result "$result" --b-images "$B_IMAGES" --a-fkd "$A_FKD" \
    --alignment-audit "$ALIGNMENT_AUDIT" --construction-manifest "$CONSTRUCTION_MANIFEST" \
    --teacher-dir "$TEACHER_DIR" --student-seed "$student" >> "$log" 2>&1
}

main(){
  exec 9>"$LOCK_ROOT/launcher.lock"; flock -n 9 || exit 75
  write_definition; rm -f "$STATUS_ROOT/launcher.failed" "$STATUS_ROOT/launcher.complete"
  echo "$(timestamp) started" > "$STATUS_ROOT/launcher.running"
  alignment_gate; echo "$(timestamp) exact alignment gate complete" > "$STATUS_ROOT/alignment.complete"
  local pids=() index=0 student gpu
  for student in "${STUDENT_SEEDS[@]}"; do
    gpu="${GPUS[$((index%${#GPUS[@]}))]}"; eval_one "$student" "$gpu" & pids+=("$!"); index=$((index+1))
  done
  wait_all "${pids[@]}"
  python "$ROOT_DIR/CV-DD/fine_grained/summarize_cross_view_label_v2.py" --root "$EXP_ROOT" \
    --a-root "$A_ROOT" --b-root "$B_ROOT" > "$LOG_ROOT/summary.log" 2>&1
  python - "$EXP_ROOT/experiment_definition.json" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads(p.read_text()); x['status']='complete'; t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
  rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) complete" > "$STATUS_ROOT/launcher.complete"
}

trap 'rc=$?; if ((rc)); then rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) exit=$rc" > "$STATUS_ROOT/launcher.failed"; fi' EXIT
main
