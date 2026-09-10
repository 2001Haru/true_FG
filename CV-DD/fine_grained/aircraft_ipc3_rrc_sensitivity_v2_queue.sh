#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_RRC_sensitivity_v2/aircraft_ipc3_v1}"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
TEACHER_DIR="${TEACHER_DIR:-/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42}"
RANDOM_ROOT="${RANDOM_ROOT:-/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard}"
RANDOM_REFERENCE_ROOT="${RANDOM_REFERENCE_ROOT:-/linxi/dataset/FG_RandomReal_soft_v2/aircraft_v1}"
RDED_ROOT="${RDED_ROOT:-/linxi/dataset/FG_RDED_original_v2/aircraft_v1}"
GPUS=(${GPUS:-0 1})
STUDENT_SEEDS=(42 43 44)
LOG_ROOT="$EXP_ROOT/logs"; STATUS_ROOT="$EXP_ROOT/status"; LOCK_ROOT="$EXP_ROOT/locks"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$LOCK_ROOT" "$EXP_ROOT/summary"

timestamp(){ date --iso-8601=seconds; }
wait_all(){ local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; ((failed==0)); }
source_images(){ if [[ "$1" == random_real ]]; then echo "$RANDOM_ROOT/selected/A_imsize224/rseed0/ipc3"; else echo "$RDED_ROOT/generated/A_imsize224/gseed42/ipc3"; fi; }
source_manifest(){ if [[ "$1" == random_real ]]; then echo "$RANDOM_ROOT/manifests/A_imsize224/rseed0/ipc3.json"; else echo "$RDED_ROOT/generated/A_imsize224/gseed42/generation_manifest.json"; fi; }

write_definition(){
python - "$EXP_ROOT" "$RANDOM_ROOT" "$RDED_ROOT" "$TEACHER_DIR" "$(git -C "$ROOT_DIR" rev-parse HEAD)" <<'PY'
import json,os,sys
from pathlib import Path
root,random_root,rded_root,teacher=map(Path,sys.argv[1:5]); revision=sys.argv[5]
expected=[str((root/'results'/source/f'ipc3_sseed{seed}.json').resolve()) for source in ('random_real','rded') for seed in (42,43,44)]
x={'status':'running','experiment':'aircraft_ipc3_rrc_scale_soft_v2','git_revision':revision,
   'dataset':'A_imsize224','ipc':3,'teacher_seed':42,'student_seeds':[42,43,44],
   'sources':{'random_real':{'selection_seed':0,'root':str(random_root.resolve())},
              'rded':{'generation_seed':42,'root':str(rded_root.resolve())}},
   'reference_rrc_scale':[0.08,1.0],'new_rrc_scale':[0.5,1.0],
   'teacher_dir':str(teacher.resolve()),'new_fkd_sets':2,'new_student_trainings':6,
   'student_protocol':'standard_protocol_v2','expected_result_files':expected,
   'summary':str((root/'summary/rrc_sensitivity_v2.json').resolve())}
p=root/'experiment_definition.json'; t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
}

relabel_one(){
  local source="$1" gpu="$2" images manifest base actual count=0 expected=6000
  images="$(source_images "$source")"; manifest="$(source_manifest "$source")"
  base="$EXP_ROOT/fkd/$source/ipc3_rrc05"; actual="${base}_bs20_ipc3"
  [[ -d "$images" && -f "$manifest" && -f "$TEACHER_DIR/ResNet18.pth" ]]
  [[ -d "$actual" ]] && count="$(find "$actual" -type f -name 'batch_*.tar' | wc -l)"
  if (( count != expected )); then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      python -u "$ROOT_DIR/CV-DD/relabel/relabel.py" --syn-data-path "$images" \
        --fkd-path "$base" --model-pool-dir "$TEACHER_DIR" --teacher-model-name ResNet18 \
        --gpu 0 --batch-size 20 --workers 8 --dataset-name A_imsize224 --epochs 400 \
        --seed 42 --fkd-seed 42 --min-scale-crops 0.5 --max-scale-crops 1 \
        --use-fp16 --mode fkd_save --mix-type cutmix > "$LOG_ROOT/relabel_${source}.log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$actual" \
    --images 300 --classes 100 --batch-size 20 --epochs 400 >> "$LOG_ROOT/relabel_${source}.log" 2>&1
}

eval_one(){
  local source="$1" student="$2" gpu="$3" images manifest fkd result log
  images="$(source_images "$source")"; manifest="$(source_manifest "$source")"
  fkd="$EXP_ROOT/fkd/$source/ipc3_rrc05_bs20_ipc3"
  result="$EXP_ROOT/results/$source/ipc3_sseed${student}.json"
  log="$LOG_ROOT/eval_${source}_sseed${student}.log"
  mkdir -p "$(dirname "$result")"; exec 7>"${result}.lock"; flock -n 7 || return 75
  if [[ ! -f "$result" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -u "$ROOT_DIR/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 \
        --exp-name "rrc05_${source}_A_ipc3_s${student}" --original-data-path "$images" \
        --fkd-path "$fkd" --output-dir "$EXP_ROOT/post_eval" --batch-size 20 --epochs 400 \
        --dataset-name A_imsize224 --gradient-accumulation-steps 2 --mix-type cutmix \
        --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$student" --temperature 20 \
        --student-initialization imagenet-v1 --student-protocol-name standard_protocol_v2 \
        --adamw-weight-decay 1e-5 --adamw-beta1 0.9 --adamw-beta2 0.999 --adamw-eps 1e-8 \
        --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 \
        --val-dir "$DATA_ROOT/A_imsize224/test" --disable-wandb --per-class-output "$result" \
        > "$log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/record_rrc_sensitivity_v2_result.py" \
    --result "$result" --source-kind "$source" --source-manifest "$manifest" \
    --source-images "$images" --teacher-dir "$TEACHER_DIR" --fkd-dir "$fkd" \
    --student-seed "$student" >> "$log" 2>&1
}

main(){
  exec 9>"$LOCK_ROOT/launcher.lock"; flock -n 9 || exit 75
  write_definition; rm -f "$STATUS_ROOT/launcher.failed" "$STATUS_ROOT/launcher.complete"
  echo "$(timestamp) started" > "$STATUS_ROOT/launcher.running"
  local pids=() index=0 source student gpu
  for source in random_real rded; do
    gpu="${GPUS[$index]}"; relabel_one "$source" "$gpu" & pids+=("$!"); index=$((index+1))
  done
  wait_all "${pids[@]}"; echo "$(timestamp) relabel complete" > "$STATUS_ROOT/relabel.complete"
  pids=(); index=0
  for source in random_real rded; do for student in "${STUDENT_SEEDS[@]}"; do
    gpu="${GPUS[$((index%${#GPUS[@]}))]}"; eval_one "$source" "$student" "$gpu" & pids+=("$!"); index=$((index+1))
  done; done
  wait_all "${pids[@]}"
  python "$ROOT_DIR/CV-DD/fine_grained/summarize_rrc_sensitivity_v2.py" --root "$EXP_ROOT" \
    --random-reference-root "$RANDOM_REFERENCE_ROOT" --rded-reference-root "$RDED_ROOT" \
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
