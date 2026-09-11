#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_DINO_centroid_soft_v2/aircraft_v1}"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
TEACHER_DIR="${TEACHER_DIR:-/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42}"
TOP5_MANIFEST="${TOP5_MANIFEST:-/linxi/dataset/FG_HardLabel_standard/v1/dino_ipc5_random_top5_kmeans5/manifests/A_imsize224/global_center_top5.json}"
IPC1_MANIFEST="${IPC1_MANIFEST:-/linxi/dataset/FG_HardLabel_standard/v1/dino_fivearm_ipc1/selections/A_imsize224/manifests/centroid.json}"
GPUS=(${GPUS:-0 1}); EVALS_PER_GPU="${EVALS_PER_GPU:-4}"
IPCS=(1 3 5); STUDENT_SEEDS=(42 43 44)
LOG_ROOT="$EXP_ROOT/logs"; STATUS_ROOT="$EXP_ROOT/status"; LOCK_ROOT="$EXP_ROOT/locks"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$LOCK_ROOT" "$EXP_ROOT/summary"

timestamp(){ date --iso-8601=seconds; }
wait_all(){ local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; ((failed==0)); }
images(){ echo "$EXP_ROOT/selection/ipc$1"; }
manifest(){ echo "$EXP_ROOT/selection/manifests/ipc$1.json"; }
fkd(){ echo "$EXP_ROOT/fkd/ipc${1}_bs20_ipc${1}"; }

write_definition(){
python - "$EXP_ROOT" "$TOP5_MANIFEST" "$IPC1_MANIFEST" "$TEACHER_DIR" "$(git -C "$ROOT_DIR" rev-parse HEAD)" <<'PY'
import json,os,sys
from pathlib import Path
root,top5,ipc1,teacher=map(Path,sys.argv[1:5]); revision=sys.argv[5]
x={'status':'running','experiment':'aircraft_dino_centroid_soft_v2','git_revision':revision,
   'dataset':'A_imsize224','method':'DINOv2 normalized class-centroid cosine Top IPC',
   'ipcs':[1,3,5],'student_seeds':[42,43,44],'teacher_seed':42,'selection_seed':None,
   'top5_manifest':str(top5.resolve()),'ipc1_manifest':str(ipc1.resolve()),'teacher_dir':str(teacher.resolve()),
   'nested_ipc':True,'expected_fkd_sets':3,'expected_results':9,'student_protocol':'standard_protocol_v2',
   'expected_result_files':[str((root/f'results/ipc{ipc}_sseed{seed}.json').resolve()) for ipc in (1,3,5) for seed in (42,43,44)],
   'summary':str((root/'summary/dino_centroid_soft_v2.json').resolve())}
p=root/'experiment_definition.json'; t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
}

prepare(){
  python "$ROOT_DIR/CV-DD/fine_grained/prepare_dino_centroid_nested.py" \
    --top5-manifest "$TOP5_MANIFEST" --ipc1-manifest "$IPC1_MANIFEST" \
    --output-root "$EXP_ROOT/selection" > "$LOG_ROOT/selection.log" 2>&1
}

relabel_one(){
  local ipc="$1" gpu="$2" base="$EXP_ROOT/fkd/ipc${ipc}" actual expected count=0
  actual="$(fkd "$ipc")"; expected=$((400*100*ipc/20))
  [[ -d "$actual" ]] && count="$(find "$actual" -type f -name 'batch_*.tar' | wc -l)"
  if (( count != expected )); then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      python -u "$ROOT_DIR/CV-DD/relabel/relabel.py" --syn-data-path "$(images "$ipc")" \
        --fkd-path "$base" --model-pool-dir "$TEACHER_DIR" --teacher-model-name ResNet18 \
        --gpu 0 --batch-size 20 --workers 8 --dataset-name A_imsize224 --epochs 400 \
        --seed 42 --fkd-seed 42 --min-scale-crops 0.08 --max-scale-crops 1 \
        --use-fp16 --mode fkd_save --mix-type cutmix > "$LOG_ROOT/relabel_ipc${ipc}.log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$actual" \
    --images $((100*ipc)) --classes 100 --batch-size 20 --epochs 400 >> "$LOG_ROOT/relabel_ipc${ipc}.log" 2>&1
}

eval_one(){
  local ipc="$1" seed="$2" gpu="$3" result="$EXP_ROOT/results/ipc${ipc}_sseed${seed}.json"
  local log="$LOG_ROOT/eval_ipc${ipc}_sseed${seed}.log"; mkdir -p "$(dirname "$result")"
  exec 7>"${result}.lock"; flock -n 7 || return 75
  if [[ ! -f "$result" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -u "$ROOT_DIR/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc "$ipc" \
        --exp-name "aircraft_dino_centroid_v2_ipc${ipc}_s${seed}" --original-data-path "$(images "$ipc")" \
        --fkd-path "$(fkd "$ipc")" --output-dir "$EXP_ROOT/post_eval" --batch-size 20 --epochs 400 \
        --dataset-name A_imsize224 --gradient-accumulation-steps 2 --mix-type cutmix --workers 8 \
        --persistent-workers --fkd_seed 42 --train-seed "$seed" --temperature 20 \
        --student-initialization imagenet-v1 --student-protocol-name standard_protocol_v2 \
        --adamw-weight-decay 1e-5 --adamw-beta1 0.9 --adamw-beta2 0.999 --adamw-eps 1e-8 \
        --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 \
        --val-dir "$DATA_ROOT/A_imsize224/test" --disable-wandb --per-class-output "$result" > "$log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/record_centroid_soft_v2_result.py" --result "$result" \
    --manifest "$(manifest "$ipc")" --fkd "$(fkd "$ipc")" --teacher "$TEACHER_DIR/ResNet18.pth" \
    --ipc "$ipc" --student-seed "$seed" >> "$log" 2>&1
}

main(){
  exec 9>"$LOCK_ROOT/launcher.lock"; flock -n 9 || exit 75
  write_definition; rm -f "$STATUS_ROOT/launcher.failed" "$STATUS_ROOT/launcher.complete"
  echo "$(timestamp) started" > "$STATUS_ROOT/launcher.running"
  prepare; echo "$(timestamp) complete" > "$STATUS_ROOT/selection.complete"
  local pids=() index=0 ipc gpu seed
  for ipc in "${IPCS[@]}"; do
    gpu="${GPUS[$((index%${#GPUS[@]}))]}"; relabel_one "$ipc" "$gpu" & pids+=("$!"); index=$((index+1))
    if (( ${#pids[@]} == ${#GPUS[@]} )); then wait_all "${pids[@]}"; pids=(); fi
  done
  if (( ${#pids[@]} )); then wait_all "${pids[@]}"; fi
  echo "$(timestamp) complete" > "$STATUS_ROOT/relabel.complete"
  pids=(); index=0
  for ipc in "${IPCS[@]}"; do for seed in "${STUDENT_SEEDS[@]}"; do
    gpu="${GPUS[$((index%${#GPUS[@]}))]}"; eval_one "$ipc" "$seed" "$gpu" & pids+=("$!"); index=$((index+1))
    if (( ${#pids[@]} == ${#GPUS[@]}*EVALS_PER_GPU )); then wait_all "${pids[@]}"; pids=(); fi
  done; done
  if (( ${#pids[@]} )); then wait_all "${pids[@]}"; fi
  python "$ROOT_DIR/CV-DD/fine_grained/summarize_centroid_soft_v2.py" --root "$EXP_ROOT" > "$LOG_ROOT/summary.log" 2>&1
  python - "$EXP_ROOT/experiment_definition.json" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads(p.read_text()); x['status']='complete'; t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
  rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) complete" > "$STATUS_ROOT/launcher.complete"
}

trap 'rc=$?; if ((rc)); then rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) exit=$rc" > "$STATUS_ROOT/launcher.failed"; fi' EXIT
main
