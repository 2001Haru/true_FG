#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_StudentInit_protocol_ablation/aircraft_h20_ipc5_v1}"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
H20_ROOT="${H20_ROOT:-/linxi/dataset/FG_TeacherScore_selection_v2/aircraft_v1}"
PACKAGING_ROOT="${PACKAGING_ROOT:-/linxi/dataset/FG_H20_packaging_v2/aircraft_ipc5_v1}"
CROSS_VIEW_ROOT="${CROSS_VIEW_ROOT:-/linxi/dataset/FG_H20_cross_view_labels_v2/aircraft_ipc5_Bimage_Alabel_v1}"
A_IMAGES="$H20_ROOT/selection/selected/entropy_t20/ipc5"
B_IMAGES="$PACKAGING_ROOT/construction/downsample_up"
A_FKD="$H20_ROOT/fkd/entropy_t20/ipc5_bs20_ipc5"
ALIGNMENT_AUDIT="$CROSS_VIEW_ROOT/preflight/fkd_alignment.json"
GPUS=(${GPUS:-0 1})
PROTOCOLS=(random_v2_lrs standard_v1)
VIEWS=(a_image_a_label b_image_a_label)
STUDENT_SEEDS=(42 43 44)
LOG_ROOT="$EXP_ROOT/logs"; STATUS_ROOT="$EXP_ROOT/status"; LOCK_ROOT="$EXP_ROOT/locks"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$LOCK_ROOT" "$EXP_ROOT/summary"

timestamp(){ date --iso-8601=seconds; }
wait_all(){ local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; ((failed==0)); }
image_root(){ if [[ "$1" == a_image_a_label ]]; then echo "$A_IMAGES"; else echo "$B_IMAGES"; fi; }

write_definition(){
python - "$EXP_ROOT" "$A_IMAGES" "$B_IMAGES" "$A_FKD" "$ALIGNMENT_AUDIT" "$(git -C "$ROOT_DIR" rev-parse HEAD)" <<'PY'
import json,os,sys
from pathlib import Path
root,a,b,fkd,audit=map(Path,sys.argv[1:6]); revision=sys.argv[6]
tasks=[(p,v,s) for p in ('random_v2_lrs','standard_v1') for v in ('a_image_a_label','b_image_a_label') for s in (42,43,44)]
x={'status':'running','experiment':'aircraft_h20_ipc5_student_initialization_and_v1_protocol','git_revision':revision,
   'dataset':'A_imsize224','ipc':5,'teacher_seed':42,'student_seeds':[42,43,44],
   'a_images':str(a.resolve()),'b_images':str(b.resolve()),'common_a_fkd':str(fkd.resolve()),
   'alignment_audit':str(audit.resolve()),'new_fkd_sets':0,'new_student_trainings':len(tasks),
   'protocols':{'random_v2_lrs':{'initialization':'random','backbone_lr':1e-4,'head_lr':1e-3,'scheduler':'full cosine 400'},
                'standard_v1':{'initialization':'random','uniform_lr':1e-3,'scheduler':'eta2 half cosine 400'}},
   'expected_result_files':[str((root/f'results/{p}/{v}/ipc5_sseed{s}.json').resolve()) for p,v,s in tasks],
   'summary':str((root/'summary/student_init_protocol_ablation.json').resolve())}
p=root/'experiment_definition.json'; t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
}

eval_one(){
  local protocol="$1" view="$2" student="$3" gpu="$4"
  local images result log
  images="$(image_root "$view")"
  result="$EXP_ROOT/results/$protocol/$view/ipc5_sseed${student}.json"
  log="$LOG_ROOT/eval_${protocol}_${view}_sseed${student}.log"
  mkdir -p "$(dirname "$result")"; exec 7>"${result}.lock"; flock -n 7 || return 75
  local protocol_args=()
  if [[ "$protocol" == random_v2_lrs ]]; then
    protocol_args=(--student-protocol-name standard_protocol_v2_random_init_only
      --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0)
  else
    protocol_args=(--student-protocol-name standard_protocol_v1
      --adamw-lr-override 1e-3 --cos --eta-override 2)
  fi
  if [[ ! -f "$result" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -u "$ROOT_DIR/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 5 \
        --exp-name "${protocol}_${view}_A_ipc5_s${student}" --original-data-path "$images" \
        --fkd-path "$A_FKD" --output-dir "$EXP_ROOT/post_eval" --batch-size 20 --epochs 400 \
        --dataset-name A_imsize224 --gradient-accumulation-steps 2 --mix-type cutmix \
        --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$student" --temperature 20 \
        --student-initialization random --adamw-weight-decay 1e-5 \
        --adamw-beta1 0.9 --adamw-beta2 0.999 --adamw-eps 1e-8 "${protocol_args[@]}" \
        --val-dir "$DATA_ROOT/A_imsize224/test" --disable-wandb --per-class-output "$result" \
        > "$log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/record_student_init_protocol_ablation.py" \
    --result "$result" --protocol "$protocol" --view "$view" --image-root "$images" \
    --a-fkd "$A_FKD" --alignment-audit "$ALIGNMENT_AUDIT" --student-seed "$student" \
    >> "$log" 2>&1
}

main(){
  exec 9>"$LOCK_ROOT/launcher.lock"; flock -n 9 || exit 75
  write_definition
  python - "$ALIGNMENT_AUDIT" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['status']=='complete' and x['total_mismatches']==0
PY
  rm -f "$STATUS_ROOT/launcher.failed" "$STATUS_ROOT/launcher.complete"
  echo "$(timestamp) started" > "$STATUS_ROOT/launcher.running"
  local pids=() index=0 protocol view student gpu
  for protocol in "${PROTOCOLS[@]}"; do for view in "${VIEWS[@]}"; do for student in "${STUDENT_SEEDS[@]}"; do
    gpu="${GPUS[$((index%${#GPUS[@]}))]}"; eval_one "$protocol" "$view" "$student" "$gpu" & pids+=("$!"); index=$((index+1))
    if (( ${#pids[@]} == ${#GPUS[@]}*3 )); then wait_all "${pids[@]}"; pids=(); fi
  done; done; done
  if (( ${#pids[@]} )); then wait_all "${pids[@]}"; fi
  python "$ROOT_DIR/CV-DD/fine_grained/summarize_student_init_protocol_ablation.py" \
    --root "$EXP_ROOT" --a-root "$H20_ROOT" --cross-view-root "$CROSS_VIEW_ROOT" \
    > "$LOG_ROOT/summary.log" 2>&1
  python - "$EXP_ROOT/experiment_definition.json" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads(p.read_text()); x['status']='complete'; t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
  rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) complete" > "$STATUS_ROOT/launcher.complete"
}

trap 'rc=$?; if ((rc)); then rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) exit=$rc" > "$STATUS_ROOT/launcher.failed"; fi' EXIT
main
