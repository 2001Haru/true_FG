#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_HardReplay_standard/v2/aircraft_v1}"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
RANDOM_ROOT="/linxi/dataset/FG_RandomReal_soft_v2/aircraft_v1"
RDED_ROOT="/linxi/dataset/FG_RDED_original_v2/aircraft_v1"
H20_ROOT="/linxi/dataset/FG_TeacherScore_selection_v2/aircraft_v1"
CENTROID_ROOT="/linxi/dataset/FG_DINO_centroid_soft_v2/aircraft_v1"
PACK_ROOT="/linxi/dataset/FG_H20_packaging_v2/aircraft_ipc5_v1"
GPUS=(${GPUS:-0 1}); EVALS_PER_GPU="${EVALS_PER_GPU:-4}"
LOG_ROOT="$EXP_ROOT/logs"; STATUS_ROOT="$EXP_ROOT/status"; LOCK_ROOT="$EXP_ROOT/locks"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$LOCK_ROOT" "$EXP_ROOT/summary"

timestamp(){ date --iso-8601=seconds; }
wait_all(){ local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; ((failed==0)); }

soft_result(){
  local method="$1" source_seed="$2" ipc="$3" student="$4"
  case "$method" in
    random_real) echo "$RANDOM_ROOT/results/A_imsize224/rseed${source_seed}/ipc${ipc}_sseed${student}.json" ;;
    original_rded) echo "$RDED_ROOT/results/A_imsize224/gseed${source_seed}/ipc${ipc}_sseed${student}.json" ;;
    h20) echo "$H20_ROOT/results/entropy_t20/ipc${ipc}_sseed${student}.json" ;;
    dino_centroid) echo "$CENTROID_ROOT/results/ipc${ipc}_sseed${student}.json" ;;
    h20_downsample_b) echo "$PACK_ROOT/results/downsample_up/ipc5_sseed${student}.json" ;;
    h20_mosaic_c) echo "$PACK_ROOT/results/same_source_mosaic/ipc5_sseed${student}.json" ;;
    *) return 2 ;;
  esac
}

result_path(){
  local method="$1" source_seed="$2" ipc="$3" student="$4" suffix=""
  [[ "$source_seed" != none ]] && suffix="/source_seed${source_seed}"
  echo "$EXP_ROOT/results/${method}${suffix}/ipc${ipc}_sseed${student}.json"
}

write_definition(){
python - "$EXP_ROOT" "$(git -C "$ROOT_DIR" rev-parse HEAD)" <<'PY'
import json,os,sys
from pathlib import Path
root=Path(sys.argv[1]); revision=sys.argv[2]; expected=[]
for method,seeds,ipcs in (
 ('random_real',(0,1,2),(1,3,5)),('original_rded',(42,43,44),(1,3,5)),
 ('h20',(None,),(1,3,5)),('dino_centroid',(None,),(1,3,5)),
 ('h20_downsample_b',(None,),(5,)),('h20_mosaic_c',(None,),(5,))):
 for source_seed in seeds:
  suffix='' if source_seed is None else f'/source_seed{source_seed}'
  for ipc in ipcs:
   for student in (42,43,44): expected.append(str((root/f'results/{method}{suffix}/ipc{ipc}_sseed{student}.json').resolve()))
x={'status':'running','experiment':'aircraft_standard_v2_hard_label_fkd_replay','git_revision':revision,
   'dataset':'A_imsize224','methods':['random_real','original_rded','h20','dino_centroid','h20_downsample_b','h20_mosaic_c'],
   'source_randomness':{'random_real':[0,1,2],'original_rded':[42,43,44],'others':None},
   'ipcs':[1,3,5],'student_seeds':[42,43,44],'unique_results':78,'logical_results_with_reused_h20_A':81,
   'reused_fkd_sets':26,'new_fkd_sets':0,'hard_supervision':'ground-truth CutMix CE weighted by actual bbox area',
   'invariants':'exact source ImageFolder, FKD sampler, RRC, flip, CutMix pairing/bbox, Student init/optimizer/scheduler/batch/epochs',
   'expected_result_files':expected,'summary':str((root/'summary/aircraft_hard_replay_v2.json').resolve())}
p=root/'experiment_definition.json'; t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
}

eval_one(){
  local method="$1" source_seed="$2" ipc="$3" student="$4" gpu="$5"
  local source result log images fkd seed_arg
  source="$(soft_result "$method" "$source_seed" "$ipc" "$student")"
  result="$(result_path "$method" "$source_seed" "$ipc" "$student")"
  log="$LOG_ROOT/${method}_src${source_seed}_ipc${ipc}_sseed${student}.log"
  [[ -f "$source" ]]; mkdir -p "$(dirname "$result")"; exec 7>"${result}.lock"; flock -n 7 || return 75
  IFS=$'\t' read -r images fkd < <(python - "$source" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); print(x['synthetic_data_path']+'\t'+x['fkd_path'])
PY
  )
  [[ -d "$images" && -d "$fkd" ]]
  if [[ ! -f "$result" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -u "$ROOT_DIR/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc "$ipc" \
        --exp-name "hard_v2_${method}_src${source_seed}_ipc${ipc}_s${student}" \
        --original-data-path "$images" --fkd-path "$fkd" --output-dir "$EXP_ROOT/post_eval" \
        --batch-size 20 --epochs 400 --dataset-name A_imsize224 --gradient-accumulation-steps 2 \
        --mix-type cutmix --fkd-hard-label --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$student" \
        --temperature 20 --student-initialization imagenet-v1 --student-protocol-name standard_protocol_v2_hard_replay \
        --adamw-weight-decay 1e-5 --adamw-beta1 0.9 --adamw-beta2 0.999 --adamw-eps 1e-8 \
        --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 \
        --val-dir "$DATA_ROOT/A_imsize224/test" --disable-wandb --per-class-output "$result" > "$log" 2>&1
  fi
  seed_arg="$source_seed"
  python "$ROOT_DIR/CV-DD/fine_grained/record_hard_replay_v2_result.py" --result "$result" \
    --source-soft-result "$source" --method "$method" --source-seed "$seed_arg" \
    --ipc "$ipc" --student-seed "$student" >> "$log" 2>&1
}

main(){
  exec 9>"$LOCK_ROOT/launcher.lock"; flock -n 9 || exit 75
  write_definition; rm -f "$STATUS_ROOT/launcher.failed" "$STATUS_ROOT/launcher.complete"
  echo "$(timestamp) started" > "$STATUS_ROOT/launcher.running"
  local pids=() index=0 method source_seed ipc student gpu
  for method in random_real original_rded h20 dino_centroid h20_downsample_b h20_mosaic_c; do
    case "$method" in
      random_real) source_seeds=(0 1 2); method_ipcs=(1 3 5) ;;
      original_rded) source_seeds=(42 43 44); method_ipcs=(1 3 5) ;;
      h20|dino_centroid) source_seeds=(none); method_ipcs=(1 3 5) ;;
      *) source_seeds=(none); method_ipcs=(5) ;;
    esac
    for source_seed in "${source_seeds[@]}"; do for ipc in "${method_ipcs[@]}"; do for student in 42 43 44; do
      gpu="${GPUS[$((index%${#GPUS[@]}))]}"; eval_one "$method" "$source_seed" "$ipc" "$student" "$gpu" & pids+=("$!"); index=$((index+1))
      if (( ${#pids[@]} == ${#GPUS[@]}*EVALS_PER_GPU )); then wait_all "${pids[@]}"; pids=(); fi
    done; done; done
  done
  if (( ${#pids[@]} )); then wait_all "${pids[@]}"; fi
  python "$ROOT_DIR/CV-DD/fine_grained/summarize_aircraft_hard_replay_v2.py" --root "$EXP_ROOT" > "$LOG_ROOT/summary.log" 2>&1
  python - "$EXP_ROOT/experiment_definition.json" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads(p.read_text()); x['status']='complete'; t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
  rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) complete" > "$STATUS_ROOT/launcher.complete"
}

trap 'rc=$?; if ((rc)); then rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) exit=$rc" > "$STATUS_ROOT/launcher.failed"; fi' EXIT
main
