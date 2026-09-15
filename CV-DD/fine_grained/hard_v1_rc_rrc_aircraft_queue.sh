#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${EXP_ROOT:-/linxi/dataset/FG_HardLabel_standard/v1/aircraft_ipc3_rc_rrc_regions_v1}"
DATA=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224
WEIGHTS=/linxi/models/torchvision/resnet18-f37072fd.pth
R0=/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard/selected/A_imsize224/rseed0/ipc3
REGIONS=/linxi/dataset/FGDD_DeCO_style/aircraft_ipc3_seed0_v1/construction/selected
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$EXP"/{logs,status,locks,results,checkpoints,summary}
exec 9>"$EXP/locks/launcher.lock";flock -n 9||exit 75
trap 's=$?;if((s));then rm -f "$EXP/status/running";echo "$(date --iso-8601=seconds) exit=$s">"$EXP/status/failed";fi' EXIT
rm -f "$EXP/status/failed" "$EXP/status/complete";date --iso-8601=seconds>"$EXP/status/running"

image_root(){ case "$1" in r0) echo "$R0";; random_regions) echo "$REGIONS/random_regions/ipc3";; fg_regions) echo "$REGIONS/fg_regions/ipc3";; *) return 2;; esac; }
run_one(){
 local mode=$1 method=$2 seed=$3 gpu=$4 result protocol extra=()
 result="$EXP/results/$mode/$method/sseed$seed.json";protocol="hard_label_v1_${mode}"
 mkdir -p "$(dirname "$result")" "$EXP/checkpoints/$mode/$method/sseed$seed"
 [[ "$mode" == rrc ]] && extra=(--train-crop-mode rrc --rrc-min-scale .08 --rrc-max-scale 1) || extra=(--train-crop-mode mild_rc)
 if [[ ! -f "$result" ]];then
  CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_hard_label_v1.py" --train-dir "$(image_root "$method")" \
   --val-dir "$DATA/test" --dataset-name A_imsize224 --num-classes 100 --ipc 3 --student-seed "$seed" \
   --result "$result" --checkpoint-dir "$EXP/checkpoints/$mode/$method/sseed$seed" --imagenet-weights-path "$WEIGHTS" \
   --total-updates 3000 --batch-size 64 --backbone-lr 3e-4 --head-lr 3e-3 --backbone-min-lr 0 --head-min-lr 0 \
   --momentum .9 --weight-decay 5e-4 --eval-every-updates 300 --workers 8 --persistent-workers --val-batch-size 256 \
   --protocol-name "$protocol" "${extra[@]}" > "$EXP/logs/eval_${mode}_${method}_s${seed}.log" 2>&1
 fi
 python "$ROOT/CV-DD/fine_grained/audit_hard_label_v1_result.py" --result "$result" --dataset A_imsize224 \
  --classes 100 --ipc 3 --student-seed "$seed" --validation-images 3333 --protocol-name "$protocol" \
  --train-crop-mode "$mode" --rrc-min-scale .08 --rrc-max-scale 1 >> "$EXP/logs/eval_${mode}_${method}_s${seed}.log" 2>&1
}
failed=0;pids=();index=0
for mode in mild_rc rrc;do for method in r0 random_regions fg_regions;do for seed in 42 43 44;do
 run_one "$mode" "$method" "$seed" $((index%2))&pids+=("$!");index=$((index+1))
 if((${#pids[@]}==4));then for pid in "${pids[@]}";do wait "$pid"||failed=1;done;pids=();fi
done;done;done
for pid in "${pids[@]}";do wait "$pid"||failed=1;done;((failed==0))
python "$ROOT/CV-DD/fine_grained/summarize_hard_v1_rc_rrc_aircraft.py" --root "$EXP">"$EXP/logs/summary.log" 2>&1
rm -f "$EXP/status/running";date --iso-8601=seconds>"$EXP/status/complete"
