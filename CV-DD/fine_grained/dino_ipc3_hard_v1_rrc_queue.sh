#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${EXP_ROOT:-/linxi/dataset/FG_HardLabel_standard/v1/dino_ipc3_top3_kmeans3}"
R0=/linxi/dataset/FG_HardLabel_standard/v1/aircraft_ipc3_rc_rrc_regions_v1/results
DATA=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224
WEIGHTS=/linxi/models/torchvision/resnet18-f37072fd.pth
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$EXP"/{logs_rrc,status,locks,results_rrc,checkpoints_rrc,summary}
exec 9>"$EXP/locks/rrc_launcher.lock";flock -n 9||exit 75
trap 's=$?;if((s));then rm -f "$EXP/status/rrc.running";echo "$(date --iso-8601=seconds) exit=$s">"$EXP/status/rrc.failed";fi' EXIT
rm -f "$EXP/status/rrc.failed" "$EXP/status/rrc.complete";date --iso-8601=seconds>"$EXP/status/rrc.running"
for arm in spherical_kmeans3_rseed0 spherical_kmeans3_rseed1 global_center_top3;do
 p="$EXP/manifests/A_imsize224/$arm.json";[ -f "$p" ];python - "$p" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));assert x['status']=='complete' and x['ipc']==3 and len(x['images'])==300
PY
done

run_one(){
 local arm=$1 seed=$2 gpu=$3 result manifest
 result="$EXP/results_rrc/$arm/sseed$seed.json";manifest="$EXP/manifests/A_imsize224/$arm.json"
 mkdir -p "$(dirname "$result")" "$EXP/checkpoints_rrc/$arm/sseed$seed"
 if [[ ! -f "$result" ]];then CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_hard_label_v1.py" \
  --train-manifest "$manifest" --val-dir "$DATA/test" --dataset-name A_imsize224 --num-classes 100 --ipc 3 \
  --student-seed "$seed" --result "$result" --checkpoint-dir "$EXP/checkpoints_rrc/$arm/sseed$seed" \
  --imagenet-weights-path "$WEIGHTS" --total-updates 3000 --batch-size 64 --backbone-lr 3e-4 --head-lr 3e-3 \
  --backbone-min-lr 0 --head-min-lr 0 --momentum .9 --weight-decay 5e-4 --eval-every-updates 300 \
  --workers 8 --persistent-workers --val-batch-size 256 --protocol-name hard_label_v1_rrc \
  --train-crop-mode rrc --rrc-min-scale .08 --rrc-max-scale 1 > "$EXP/logs_rrc/eval_${arm}_s${seed}.log" 2>&1;fi
 python "$ROOT/CV-DD/fine_grained/audit_hard_label_v1_result.py" --result "$result" --dataset A_imsize224 \
  --classes 100 --ipc 3 --student-seed "$seed" --validation-images 3333 --protocol-name hard_label_v1_rrc \
  --train-crop-mode rrc --rrc-min-scale .08 --rrc-max-scale 1 >> "$EXP/logs_rrc/eval_${arm}_s${seed}.log" 2>&1
}
failed=0;pids=();index=0
for arm in spherical_kmeans3_rseed0 spherical_kmeans3_rseed1 global_center_top3;do for seed in 42 43 44;do
 run_one "$arm" "$seed" $((index%2))&pids+=("$!");index=$((index+1));if((${#pids[@]}==4));then for pid in "${pids[@]}";do wait "$pid"||failed=1;done;pids=();fi
done;done
for pid in "${pids[@]}";do wait "$pid"||failed=1;done;((failed==0))
python "$ROOT/CV-DD/fine_grained/summarize_dino_ipc3_hard_v1_rrc.py" --root "$EXP" --r0-root "$R0">"$EXP/logs_rrc/summary.log" 2>&1
rm -f "$EXP/status/rrc.running";date --iso-8601=seconds>"$EXP/status/rrc.complete"
