#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_RDED_Random_fullframe_v2/aircraft_ipc3_v1}"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
TEACHER_DIR="${TEACHER_DIR:-/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42}"
RANDOM_ROOT="/linxi/dataset/FG_RandomReal_soft_v2/aircraft_v1"; RDED_ROOT="/linxi/dataset/FG_RDED_original_v2/aircraft_v1"
GPUS=(${GPUS:-0 1}); EVALS_PER_GPU="${EVALS_PER_GPU:-4}"
LOG_ROOT="$EXP_ROOT/logs";STATUS_ROOT="$EXP_ROOT/status";LOCK_ROOT="$EXP_ROOT/locks";mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$LOCK_ROOT" "$EXP_ROOT/summary"
timestamp(){ date --iso-8601=seconds; };wait_all(){ local f=0 p;for p in "$@";do wait "$p"||f=1;done;((f==0)); }
source_result(){ if [[ "$1" == random_real ]];then echo "$RANDOM_ROOT/results/A_imsize224/rseed${2}/ipc3_sseed${3}.json";else echo "$RDED_ROOT/results/A_imsize224/gseed${2}/ipc3_sseed${3}.json";fi; }
fkd(){ echo "$EXP_ROOT/fkd/${1}/source_seed${2}/ipc3_bs20_ipc3"; }
image_root(){ python - "$(source_result "$1" "$2" 42)" <<'PY'
import json,sys;print(json.load(open(sys.argv[1]))['synthetic_data_path'])
PY
}
relabel_one(){
 local method="$1" src="$2" gpu="$3" base="$EXP_ROOT/fkd/${method}/source_seed${src}/ipc3" actual="$(fkd "$method" "$src")" count=0
 [[ -d "$actual" ]]&&count="$(find "$actual" -type f -name 'batch_*.tar'|wc -l)"
 if ((count!=6000));then CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  python -u "$ROOT_DIR/CV-DD/relabel/relabel.py" --syn-data-path "$(image_root "$method" "$src")" --fkd-path "$base" \
  --model-pool-dir "$TEACHER_DIR" --teacher-model-name ResNet18 --gpu 0 --batch-size 20 --workers 8 --dataset-name A_imsize224 \
  --epochs 400 --seed 42 --fkd-seed 42 --min-scale-crops 1 --max-scale-crops 1 --full-image-resize --use-fp16 --mode fkd_save \
  >"$LOG_ROOT/relabel_${method}_src${src}.log" 2>&1;fi
 python "$ROOT_DIR/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$actual" --images 300 --classes 100 --batch-size 20 --epochs 400 >>"$LOG_ROOT/relabel_${method}_src${src}.log" 2>&1
}
eval_one(){
 local label="$1" method="$2" src="$3" student="$4" gpu="$5" source="$(source_result "$method" "$src" "$student")"
 local result="$EXP_ROOT/results/${label}/${method}/source_seed${src}/ipc3_sseed${student}.json" log="$LOG_ROOT/eval_${label}_${method}_src${src}_s${student}.log"
 mkdir -p "$(dirname "$result")";exec 7>"${result}.lock";flock -n 7||return 75
 IFS=$'\t' read -r images oldfkd < <(python - "$source" <<'PY'
import json,sys;x=json.load(open(sys.argv[1]));print(x['synthetic_data_path']+'\t'+x['fkd_path'])
PY
 );local hard_args=() protocol=standard_protocol_v2
 if [[ "$label" == hard ]];then hard_args=(--fkd-hard-label);protocol=standard_protocol_v2_hard_replay;fi
 if [[ ! -f "$result" ]];then CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u "$ROOT_DIR/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 --exp-name "fullframe_${label}_${method}_${src}_s${student}" \
  --original-data-path "$images" --fkd-path "$(fkd "$method" "$src")" --output-dir "$EXP_ROOT/post_eval" --batch-size 20 --epochs 400 \
  --dataset-name A_imsize224 --gradient-accumulation-steps 2 "${hard_args[@]}" --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$student" \
  --temperature 20 --student-initialization imagenet-v1 --student-protocol-name "$protocol" --adamw-weight-decay 1e-5 --adamw-beta1 .9 --adamw-beta2 .999 --adamw-eps 1e-8 \
  --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 --val-dir "$DATA_ROOT/A_imsize224/test" --disable-wandb --per-class-output "$result" >"$log" 2>&1;fi
 if [[ "$label" == soft ]];then python "$ROOT_DIR/CV-DD/fine_grained/record_aircraft_fullframe_soft_v2_result.py" --result "$result" --source-image-result "$source" \
  --fkd "$(fkd "$method" "$src")" --teacher "$TEACHER_DIR/ResNet18.pth" --method "$method" --source-seed "$src" --student-seed "$student" >>"$log" 2>&1
 else local soft="$EXP_ROOT/results/soft/${method}/source_seed${src}/ipc3_sseed${student}.json";python "$ROOT_DIR/CV-DD/fine_grained/record_hard_replay_v2_result.py" \
  --result "$result" --source-soft-result "$soft" --method "$method" --source-seed "$src" --ipc 3 --student-seed "$student" --expected-mix-type none --require-full-image-resize >>"$log" 2>&1;fi
}
main(){
 exec 9>"$LOCK_ROOT/launcher.lock";flock -n 9||exit 75;rm -f "$STATUS_ROOT/launcher.failed" "$STATUS_ROOT/launcher.complete";echo "$(timestamp) started expected=36" >"$STATUS_ROOT/launcher.running"
 local pids=() idx=0 method src gpu label student
 for method in random_real original_rded;do if [[ "$method" == random_real ]];then seeds=(0 1 2);else seeds=(42 43 44);fi
  for src in "${seeds[@]}";do gpu="${GPUS[$((idx%2))]}";relabel_one "$method" "$src" "$gpu"&pids+=("$!");idx=$((idx+1));if((${#pids[@]}==2));then wait_all "${pids[@]}";pids=();fi;done
 done;if((${#pids[@]}));then wait_all "${pids[@]}";fi;echo "$(timestamp) complete" >"$STATUS_ROOT/relabel.complete"
 pids=();idx=0
 for label in soft hard;do for method in random_real original_rded;do if [[ "$method" == random_real ]];then seeds=(0 1 2);else seeds=(42 43 44);fi
  for src in "${seeds[@]}";do for student in 42 43 44;do gpu="${GPUS[$((idx%2))]}";eval_one "$label" "$method" "$src" "$student" "$gpu"&pids+=("$!");idx=$((idx+1));if((${#pids[@]}==2*EVALS_PER_GPU));then wait_all "${pids[@]}";pids=();fi;done;done
 done;done;if((${#pids[@]}));then wait_all "${pids[@]}";fi
 python "$ROOT_DIR/CV-DD/fine_grained/summarize_aircraft_rded_random_fullframe.py" --root "$EXP_ROOT" >"$LOG_ROOT/summary.log" 2>&1
 rm -f "$STATUS_ROOT/launcher.running";echo "$(timestamp) complete" >"$STATUS_ROOT/launcher.complete"
}
trap 'rc=$?;if((rc));then rm -f "$STATUS_ROOT/launcher.running";echo "$(timestamp) exit=$rc">"$STATUS_ROOT/launcher.failed";fi' EXIT
main
