#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."&&pwd)"
EXP_ROOT=/linxi/dataset/FGDD_v1/aircraft_ipc3_label_controls_v1
FGDD_ROOT=/linxi/dataset/FGDD_v1/aircraft_ipc3_seed0_v1
DATA_ROOT=/linxi/dataset/FG_SRe2L_repro/v1/datasets
IMAGES="$FGDD_ROOT/selection/selected/r2_ordinary_gain/ipc3"
FKD="$FGDD_ROOT/fkd/r2_ordinary_gain/ipc3_bs20_ipc3"
GPUS=(${GPUS:-0 1})
mkdir -p "$EXP_ROOT"/{logs,status,locks,results/hard_fullframe/r2_ordinary_gain,post_eval/hard_fullframe/r2_ordinary_gain,summary}
ts(){ date --iso-8601=seconds; }
eval_one(){
 local student gpu result log
 student=$1;gpu=$2
 result="$EXP_ROOT/results/hard_fullframe/r2_ordinary_gain/ipc3_sseed$student.json"
 log="$EXP_ROOT/logs/eval_hard_fullframe_r2_s$student.log"
 exec 7>"$result.lock";flock -n 7||return 75
 if [[ ! -f $result ]];then
  CUDA_VISIBLE_DEVICES=$gpu PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u "$ROOT_DIR/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 --exp-name "fgdd_hard_fullframe_r2_s$student" --original-data-path "$IMAGES" --fkd-path "$FKD" --output-dir "$EXP_ROOT/post_eval/hard_fullframe/r2_ordinary_gain/sseed$student" --batch-size 20 --epochs 400 --dataset-name A_imsize224 --gradient-accumulation-steps 2 --fkd-hard-label --student-protocol-name standard_protocol_v2_fgdd_hard_fullframe --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$student" --temperature 20 --student-initialization imagenet-v1 --adamw-weight-decay 1e-5 --adamw-beta1 .9 --adamw-beta2 .999 --adamw-eps 1e-8 --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 --val-dir "$DATA_ROOT/A_imsize224/test" --disable-wandb --per-class-output "$result">"$log" 2>&1
 fi
}
main(){
 exec 9>"$EXP_ROOT/locks/r2_hard_launcher.lock";flock -n 9||exit 75
 rm -f "$EXP_ROOT/status/r2_hard.failed" "$EXP_ROOT/status/r2_hard.complete"
 echo "$(ts) started expected=3">"$EXP_ROOT/status/r2_hard.running"
 local pids=() i=0 student gpu p failed=0
 for student in 42 43 44;do gpu=${GPUS[$((i%2))]};eval_one "$student" "$gpu"&pids+=("$!");i=$((i+1));done
 for p in "${pids[@]}";do wait "$p"||failed=1;done;((failed==0))
 python "$ROOT_DIR/CV-DD/fine_grained/summarize_fgdd_v1_r2_hard.py" --root "$EXP_ROOT">"$EXP_ROOT/logs/r2_hard_summary.log" 2>&1
 rm -f "$EXP_ROOT/status/r2_hard.running";echo "$(ts) complete">"$EXP_ROOT/status/r2_hard.complete"
}
trap 'r=$?;if((r));then rm -f "$EXP_ROOT/status/r2_hard.running";echo "$(ts) exit=$r">"$EXP_ROOT/status/r2_hard.failed";fi' EXIT
main
