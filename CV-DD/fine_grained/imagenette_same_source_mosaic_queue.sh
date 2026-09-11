#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_ImageNette_same_source_mosaic/v1}"
AB_ROOT="${AB_ROOT:-/linxi/dataset/FG_ImageNette_native_resolution/v1}"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/CV-DD/test_data/imagenet-nette}"
TEACHER="${TEACHER:-/linxi/dataset/CV-DD/offline_models/imagenet-nette/ResNet18.pth}"
CONSTRUCTION_ROOT="$EXP_ROOT/construction"
CONSTRUCTION_MANIFEST="$CONSTRUCTION_ROOT/construction_manifest.json"
GPUS=(${GPUS:-0 1})
IPCS=(10 50)
STUDENT_SEEDS=(42 43 44)
LOG_ROOT="$EXP_ROOT/logs"; STATUS_ROOT="$EXP_ROOT/status"; LOCK_ROOT="$EXP_ROOT/locks"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$LOCK_ROOT" "$EXP_ROOT/summary"

timestamp(){ date --iso-8601=seconds; }
wait_all(){ local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; ((failed==0)); }
a_fkd(){ echo "$AB_ROOT/fkd/a_image/ipc${1}_bs10_ipc${1}"; }
c_fkd(){ echo "$EXP_ROOT/fkd/c_image/ipc${1}"; }

write_definition(){
python - "$EXP_ROOT" "$AB_ROOT" "$TEACHER" "$(git -C "$ROOT_DIR" rev-parse HEAD)" <<'PY'
import json,os,sys
from pathlib import Path
root,ab,teacher=map(Path,sys.argv[1:4]); revision=sys.argv[4]
x={'status':'running','experiment':'imagenette_same_source_mosaic_native_cvdd','git_revision':revision,
   'dataset':'imagenet-nette','ipcs':[10,50],'student_seeds':[42,43,44],
   'conditions':{'A':'existing A image/A label','B':'existing B image/A label','C':'new same-source 2x2 image/C rescored label'},
   'ab_root':str(ab.resolve()),'teacher':str(teacher.resolve()),'construction_seed':42,
   'source_occurrences_static':4,'new_fkd_sets':2,'new_student_trainings':6,
   'fkd_metadata':'exact A batch/RRC/flip/CutMix replay; C Teacher logits rescored in train-mode BSSL',
   'expected_result_files':[str((root/f'results/ipc{ipc}_sseed{seed}.json').resolve()) for ipc in (10,50) for seed in (42,43,44)],
   'summary':str((root/'summary/imagenette_same_source_mosaic.json').resolve())}
p=root/'experiment_definition.json'; t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
}

prepare(){
  python -u "$ROOT_DIR/CV-DD/fine_grained/prepare_imagenette_same_source_mosaic.py" \
    --ab-construction-manifest "$AB_ROOT/construction/construction_manifest.json" \
    --output-root "$CONSTRUCTION_ROOT" --construction-seed 42 --skip-completed \
    > "$LOG_ROOT/construction.log" 2>&1
}

rescore_one(){
  local ipc="$1" gpu="$2" source target expected count=0
  source="$(a_fkd "$ipc")"; target="$(c_fkd "$ipc")"; expected=$((300*ipc))
  [[ -d "$target" ]] && count="$(find "$target" -type f -name 'batch_*.tar' | wc -l)"
  if (( count != expected )); then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      python -u "$ROOT_DIR/CV-DD/fine_grained/rescore_fkd_views.py" \
        --image-root "$CONSTRUCTION_ROOT/ipc${ipc}" --source-fkd "$source" \
        --output-fkd "$target" --teacher "$TEACHER" --ipc "$ipc" \
        --epochs 300 --batch-size 10 --workers 2 --seed 42 --fkd-seed 42 --skip-completed \
        > "$LOG_ROOT/relabel_ipc${ipc}.log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$target" \
    --images $((10*ipc)) --classes 10 --batch-size 10 --epochs 300 >> "$LOG_ROOT/relabel_ipc${ipc}.log" 2>&1
  python "$ROOT_DIR/CV-DD/fine_grained/audit_fkd_metadata_replay.py" \
    --source-fkd "$source" --target-fkd "$target" --epochs 300 \
    --batches-per-epoch "$ipc" --batch-size 10 \
    --output "$target/metadata_replay_audit.json" >> "$LOG_ROOT/relabel_ipc${ipc}.log" 2>&1
}

eval_one(){
  local ipc="$1" student="$2" gpu="$3" eta fkd images result log
  [[ "$ipc" == 10 ]] && eta=1 || eta=2
  fkd="$(c_fkd "$ipc")"; images="$CONSTRUCTION_ROOT/ipc${ipc}"
  result="$EXP_ROOT/results/ipc${ipc}_sseed${student}.json"
  log="$LOG_ROOT/eval_ipc${ipc}_sseed${student}.log"
  mkdir -p "$(dirname "$result")"; exec 7>"${result}.lock"; flock -n 7 || return 75
  if [[ ! -f "$result" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -u "$ROOT_DIR/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc "$ipc" \
        --exp-name "imagenette_native_c_mosaic_ipc${ipc}_s${student}" \
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
  python "$ROOT_DIR/CV-DD/fine_grained/record_imagenette_mosaic_result.py" \
    --result "$result" --ipc "$ipc" --student-seed "$student" --image-root "$images" \
    --c-fkd "$fkd" --construction-manifest "$CONSTRUCTION_MANIFEST" \
    --metadata-audit "$fkd/metadata_replay_audit.json" --teacher "$TEACHER" >> "$log" 2>&1
}

main(){
  exec 9>"$LOCK_ROOT/launcher.lock"; flock -n 9 || exit 75
  write_definition; rm -f "$STATUS_ROOT/launcher.failed" "$STATUS_ROOT/launcher.complete"
  echo "$(timestamp) started" > "$STATUS_ROOT/launcher.running"
  prepare; echo "$(timestamp) construction complete" > "$STATUS_ROOT/construction.complete"
  local pids=() index=0 ipc student gpu
  for ipc in "${IPCS[@]}"; do
    gpu="${GPUS[$index]}"; rescore_one "$ipc" "$gpu" & pids+=("$!"); index=$((index+1))
  done
  wait_all "${pids[@]}"; echo "$(timestamp) C FKD rescore and metadata audit complete" > "$STATUS_ROOT/relabel.complete"
  pids=(); index=0
  for ipc in "${IPCS[@]}"; do for student in "${STUDENT_SEEDS[@]}"; do
    gpu="${GPUS[$((index%${#GPUS[@]}))]}"; eval_one "$ipc" "$student" "$gpu" & pids+=("$!"); index=$((index+1))
  done; done
  wait_all "${pids[@]}"
  python "$ROOT_DIR/CV-DD/fine_grained/summarize_imagenette_same_source_mosaic.py" \
    --root "$EXP_ROOT" --ab-root "$AB_ROOT" > "$LOG_ROOT/summary.log" 2>&1
  python - "$EXP_ROOT/experiment_definition.json" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads(p.read_text()); x['status']='complete'; t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
  rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) complete" > "$STATUS_ROOT/launcher.complete"
}

trap 'rc=$?; if ((rc)); then rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) exit=$rc" > "$STATUS_ROOT/launcher.failed"; fi' EXIT
main
