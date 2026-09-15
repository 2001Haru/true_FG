#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PARENT=/linxi/dataset/FGDD_NRR/aircraft_ipc3_pixel_optimization_v1
EXP="$PARENT/hard_v1"
CONSTRUCTION="$PARENT/construction/construction_manifest.json"
R0_ROOT=/linxi/dataset/FG_HardLabel_standard/v1/aircraft_ipc3_rc_rrc_regions_v1
TEST=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/test
WEIGHTS=/linxi/models/torchvision/resnet18-f37072fd.pth
ARMS=(plain cam_protected random_protected)
MODES=(mild_rc rrc)
SEEDS=(42 43 44)
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$EXP"/{logs,status,locks,results,checkpoints,summary,audits}
exec 9>"$EXP/locks/launcher.lock"; flock -n 9 || exit 75
trap 's=$?; if ((s)); then rm -f "$EXP/status/running"; echo "$(date --iso-8601=seconds) exit=$s" > "$EXP/status/failed"; fi' EXIT
rm -f "$EXP/status/failed" "$EXP/status/complete"; date --iso-8601=seconds > "$EXP/status/running"

python - "$CONSTRUCTION" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);x=json.loads(p.read_text())
assert x['status']=='complete' and x['images_per_arm']==300
assert set(x['arms'])=={'plain','cam_protected','random_protected'}
for arm,row in x['arms'].items():
 root=Path(row['image_root']); files=sorted(root.glob('*/*.png'))
 assert len(files)==300 and len([d for d in root.iterdir() if d.is_dir()])==100
 assert max(item['protected_max_abs'] for item in json.loads(Path(row['manifest']).read_text())['exports'])==0
print('construction reuse gate passed')
PY

run_one(){
 local mode=$1 arm=$2 seed=$3 gpu=$4 result="$EXP/results/$mode/$arm/sseed$seed.json" extra=()
 mkdir -p "$(dirname "$result")" "$EXP/checkpoints/$mode/$arm/sseed$seed"
 [[ $mode == rrc ]] && extra=(--train-crop-mode rrc --rrc-min-scale .08 --rrc-max-scale 1) || extra=(--train-crop-mode mild_rc)
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_hard_label_v1.py" \
  --train-dir "$PARENT/construction/selected/$arm/ipc3" --val-dir "$TEST" --dataset-name A_imsize224 --num-classes 100 \
  --ipc 3 --student-seed "$seed" --result "$result" --checkpoint-dir "$EXP/checkpoints/$mode/$arm/sseed$seed" \
  --imagenet-weights-path "$WEIGHTS" --total-updates 3000 --batch-size 64 --backbone-lr 3e-4 --head-lr 3e-3 \
  --backbone-min-lr 0 --head-min-lr 0 --momentum .9 --weight-decay 5e-4 --eval-every-updates 300 \
  --workers 8 --persistent-workers --val-batch-size 256 --protocol-name "hard_label_v1_${mode}" "${extra[@]}" \
  > "$EXP/logs/eval_${mode}_${arm}_s${seed}.log" 2>&1
 python "$ROOT/CV-DD/fine_grained/audit_hard_label_v1_result.py" --result "$result" --dataset A_imsize224 \
  --classes 100 --ipc 3 --student-seed "$seed" --validation-images 3333 --protocol-name "hard_label_v1_${mode}" \
  --train-crop-mode "$mode" --rrc-min-scale .08 --rrc-max-scale 1 >> "$EXP/logs/eval_${mode}_${arm}_s${seed}.log" 2>&1
}
failed=0; pids=(); index=0
for mode in "${MODES[@]}"; do for arm in "${ARMS[@]}"; do for seed in "${SEEDS[@]}"; do
 run_one "$mode" "$arm" "$seed" $((index%2)) & pids+=("$!"); index=$((index+1))
 if ((${#pids[@]}==4)); then for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; pids=(); fi
done; done; done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; ((failed==0))
python "$ROOT/CV-DD/fine_grained/summarize_nrr_pixel_hard_v1_rc_rrc.py" --root "$EXP" --r0-root "$R0_ROOT" \
 > "$EXP/logs/summary.log" 2>&1
rm -f "$EXP/status/running"; date --iso-8601=seconds > "$EXP/status/complete"
