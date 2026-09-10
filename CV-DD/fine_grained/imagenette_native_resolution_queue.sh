#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_ImageNette_native_resolution/v1}"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/CV-DD/test_data/imagenet-nette}"
TEACHER_DIR="${TEACHER_DIR:-/linxi/dataset/CV-DD/offline_models/imagenet-nette}"
PROTOCOL="$ROOT_DIR/CV-DD/fine_grained/imagenette_native_resolution_protocol.json"
CONSTRUCTION_ROOT="$EXP_ROOT/construction"
CONSTRUCTION_MANIFEST="$CONSTRUCTION_ROOT/construction_manifest.json"
GPUS=(${GPUS:-0 1})
IPCS=(10 50)
VIEWS=(a_image b_image)
STUDENT_SEEDS=(42 43 44)
LOG_ROOT="$EXP_ROOT/logs"; STATUS_ROOT="$EXP_ROOT/status"; LOCK_ROOT="$EXP_ROOT/locks"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$LOCK_ROOT" "$EXP_ROOT/summary"

timestamp(){ date --iso-8601=seconds; }
wait_all(){ local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; ((failed==0)); }
image_root(){ echo "$CONSTRUCTION_ROOT/selected/$1/ipc$2"; }
fkd_dir(){ echo "$EXP_ROOT/fkd/a_image/ipc${1}_bs10_ipc${1}"; }

write_definition(){
python - "$EXP_ROOT" "$DATA_ROOT" "$TEACHER_DIR" "$PROTOCOL" "$(git -C "$ROOT_DIR" rev-parse HEAD)" <<'PY'
import json,os,sys
from pathlib import Path
root,data,teacher,protocol=map(Path,sys.argv[1:5]); revision=sys.argv[5]
tasks=[(ipc,view,seed) for ipc in (10,50) for view in ('a_image','b_image') for seed in (42,43,44)]
x={'status':'prepared_not_started','experiment':'imagenette_cvdd_native_resolution','git_revision':revision,
   'dataset':'imagenet-nette','ipcs':[10,50],'views':['a_image','b_image'],'student_seeds':[42,43,44],
   'source_selection':'per-class H20 entropy ascending, deterministic nested Top10/50',
   'teacher_dir':str(teacher.resolve()),'data_root':str(data.resolve()),
   'protocol_definition':str(protocol.resolve()),'new_fkd_sets':2,'new_student_trainings':12,
   'batch_correction':'Validate batch16 -> batch10 to match paper Table24 and Relabel constants',
   'expected_result_files':[str((root/f'results/{view}/ipc{ipc}_sseed{seed}.json').resolve()) for ipc,view,seed in tasks],
   'summary':str((root/'summary/imagenette_native_resolution.json').resolve())}
p=root/'experiment_definition.json'; t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
}

prepare(){
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -u "$ROOT_DIR/CV-DD/fine_grained/prepare_imagenette_resolution_pairs.py" \
      --data-root "$DATA_ROOT" --teacher "$TEACHER_DIR/ResNet18.pth" \
      --output-root "$CONSTRUCTION_ROOT" --batch-size 256 --skip-completed \
      > "$LOG_ROOT/construction.log" 2>&1
}

relabel_one(){
  local ipc="$1" gpu="$2" images base actual expected count=0 log
  images="$(image_root a_image "$ipc")"; base="$EXP_ROOT/fkd/a_image/ipc${ipc}"
  actual="$(fkd_dir "$ipc")"; expected=$((300*ipc)); log="$LOG_ROOT/relabel_ipc${ipc}.log"
  [[ -d "$actual" ]] && count="$(find "$actual" -type f -name 'batch_*.tar' | wc -l)"
  if (( count != expected )); then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      python -u "$ROOT_DIR/CV-DD/relabel/relabel.py" --syn-data-path "$images" \
        --fkd-path "$base" --model-pool-dir "$TEACHER_DIR" --teacher-model-name ResNet18 \
        --gpu 0 --batch-size 10 --workers 2 --dataset-name imagenet-nette --epochs 300 \
        --seed 42 --fkd-seed 42 --min-scale-crops 0.08 --max-scale-crops 1 \
        --use-fp16 --mode fkd_save --mix-type cutmix > "$log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$actual" \
    --images $((10*ipc)) --classes 10 --batch-size 10 --epochs 300 >> "$log" 2>&1
}

eval_one(){
  local ipc="$1" view="$2" student="$3" gpu="$4" images fkd eta result log
  images="$(image_root "$view" "$ipc")"; fkd="$(fkd_dir "$ipc")"
  if [[ "$ipc" == 10 ]]; then eta=1; else eta=2; fi
  result="$EXP_ROOT/results/$view/ipc${ipc}_sseed${student}.json"
  log="$LOG_ROOT/eval_${view}_ipc${ipc}_sseed${student}.log"
  mkdir -p "$(dirname "$result")"; exec 7>"${result}.lock"; flock -n 7 || return 75
  if [[ ! -f "$result" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -u "$ROOT_DIR/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc "$ipc" \
        --exp-name "imagenette_native_${view}_ipc${ipc}_s${student}" \
        --original-data-path "$images" --fkd-path "$fkd" --output-dir "$EXP_ROOT/post_eval" \
        --batch-size 10 --epochs 300 --dataset-name imagenet-nette \
        --gradient-accumulation-steps 2 --mix-type cutmix --workers 2 \
        --fkd_seed 42 --train-seed "$student" --temperature 20 --student-initialization random \
        --student-protocol-name imagenette_cvdd_native_resolution_v1 \
        --adamw-lr-override 5e-4 --adamw-weight-decay 0.01 \
        --adamw-beta1 0.9 --adamw-beta2 0.999 --adamw-eps 1e-8 \
        --cos --eta-override "$eta" --val-dir "$DATA_ROOT/test" --disable-wandb \
        --per-class-output "$result" > "$log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/record_imagenette_native_resolution_result.py" \
    --result "$result" --view "$view" --ipc "$ipc" --student-seed "$student" \
    --image-root "$images" --a-fkd "$fkd" --construction-manifest "$CONSTRUCTION_MANIFEST" \
    --protocol "$PROTOCOL" --teacher-dir "$TEACHER_DIR" >> "$log" 2>&1
}

launcher(){
  exec 9>"$LOCK_ROOT/launcher.lock"; flock -n 9 || exit 75
  write_definition; rm -f "$STATUS_ROOT/launcher.failed" "$STATUS_ROOT/launcher.complete"
  echo "$(timestamp) started" > "$STATUS_ROOT/launcher.running"
  python - "$EXP_ROOT/experiment_definition.json" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads(p.read_text()); x['status']='running'; t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
  prepare; echo "$(timestamp) construction complete" > "$STATUS_ROOT/construction.complete"
  local pids=() index=0 ipc view student gpu
  for ipc in "${IPCS[@]}"; do
    gpu="${GPUS[$index]}"; relabel_one "$ipc" "$gpu" & pids+=("$!"); index=$((index+1))
  done
  wait_all "${pids[@]}"; echo "$(timestamp) relabel complete" > "$STATUS_ROOT/relabel.complete"
  pids=(); index=0
  for ipc in "${IPCS[@]}"; do for view in "${VIEWS[@]}"; do for student in "${STUDENT_SEEDS[@]}"; do
    gpu="${GPUS[$((index%${#GPUS[@]}))]}"; eval_one "$ipc" "$view" "$student" "$gpu" & pids+=("$!"); index=$((index+1))
    if (( ${#pids[@]} == ${#GPUS[@]}*3 )); then wait_all "${pids[@]}"; pids=(); fi
  done; done; done
  if (( ${#pids[@]} )); then wait_all "${pids[@]}"; fi
  python "$ROOT_DIR/CV-DD/fine_grained/summarize_imagenette_native_resolution.py" --root "$EXP_ROOT" \
    > "$LOG_ROOT/summary.log" 2>&1
  python - "$EXP_ROOT/experiment_definition.json" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads(p.read_text()); x['status']='complete'; t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
  rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) complete" > "$STATUS_ROOT/launcher.complete"
}

trap 'rc=$?; if ((rc)); then rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) exit=$rc" > "$STATUS_ROOT/launcher.failed"; fi' EXIT

case "${1:-prepare}" in
  prepare) write_definition ;;
  launch) launcher ;;
  *) echo "usage: $0 prepare|launch" >&2; exit 2 ;;
esac
