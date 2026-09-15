#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${EXP_ROOT:-/linxi/dataset/FG_HardLabel_standard/v1/dino_ipc3_top3_kmeans3}"
PARENT=/linxi/dataset/FG_HardLabel_standard/v1/dino_fivearm_ipc1
IPC5=/linxi/dataset/FG_HardLabel_standard/v1/dino_ipc5_random_top5_kmeans5
R0_RESULTS=/linxi/dataset/FG_HardLabel_standard/v1/aircraft_ipc3_rc_rrc_regions_v1/results/mild_rc/r0
DATA=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224
WEIGHTS=/linxi/models/torchvision/resnet18-f37072fd.pth
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$EXP"/{logs,status,locks,results,checkpoints,summary,manifests,selection_audits}
exec 9>"$EXP/locks/launcher.lock";flock -n 9||exit 75
trap 's=$?;if((s));then rm -f "$EXP/status/running";echo "$(date --iso-8601=seconds) exit=$s">"$EXP/status/failed";fi' EXIT
rm -f "$EXP/status/failed" "$EXP/status/complete";date --iso-8601=seconds>"$EXP/status/running"

python "$ROOT/CV-DD/fine_grained/prepare_dino_ipc3_selection.py" --parent-experiment-root "$PARENT" \
 --output-root "$EXP" --dataset-name A_imsize224 --classes 100 > "$EXP/logs/selection.log" 2>&1
python - "$EXP" "$IPC5" <<'PY'
import json,sys
from pathlib import Path
root,ipc5=map(Path,sys.argv[1:3]);audit=json.loads((root/'selection_audits/A_imsize224.json').read_text())
assert audit['status']=='complete' and audit['spherical_kmeans']['total_empty_cluster_repair_events']==0
top3=json.loads((root/'manifests/A_imsize224/global_center_top3.json').read_text())['images']
top5=json.loads((ipc5/'manifests/A_imsize224/global_center_top5.json').read_text())['images']
for c in range(100):
 a=[x['source_path'] for x in top3 if x['class_id']==c]
 b=[x['source_path'] for x in top5 if x['class_id']==c]
 assert a==b[:3] and len(set(a))==3
for seed in (0,1,2):
 rows=json.loads((root/f'manifests/A_imsize224/spherical_kmeans3_rseed{seed}.json').read_text())['images']
 for c in range(100):
  xs=[x['source_path'] for x in rows if x['class_id']==c];assert len(xs)==len(set(xs))==3
print('selection gate passed')
PY
date --iso-8601=seconds>"$EXP/status/selection.complete"

run_one(){
 local arm=$1 seed=$2 gpu=$3 result manifest
 result="$EXP/results/$arm/sseed$seed.json";manifest="$EXP/manifests/A_imsize224/$arm.json"
 mkdir -p "$(dirname "$result")" "$EXP/checkpoints/$arm/sseed$seed"
 if [[ ! -f "$result" ]];then CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_hard_label_v1.py" \
  --train-manifest "$manifest" --val-dir "$DATA/test" --dataset-name A_imsize224 --num-classes 100 --ipc 3 \
  --student-seed "$seed" --result "$result" --checkpoint-dir "$EXP/checkpoints/$arm/sseed$seed" \
  --imagenet-weights-path "$WEIGHTS" --total-updates 3000 --batch-size 64 --backbone-lr 3e-4 --head-lr 3e-3 \
  --backbone-min-lr 0 --head-min-lr 0 --momentum .9 --weight-decay 5e-4 --eval-every-updates 300 \
  --workers 8 --persistent-workers --val-batch-size 256 > "$EXP/logs/eval_${arm}_s${seed}.log" 2>&1;fi
 python "$ROOT/CV-DD/fine_grained/audit_hard_label_v1_result.py" --result "$result" --dataset A_imsize224 \
  --classes 100 --ipc 3 --student-seed "$seed" --validation-images 3333 >> "$EXP/logs/eval_${arm}_s${seed}.log" 2>&1
}
failed=0;pids=();index=0
for rseed in 0 1 2;do for seed in 42 43 44;do arm="spherical_kmeans3_rseed$rseed";run_one "$arm" "$seed" $((index%2))&pids+=("$!");index=$((index+1));if((${#pids[@]}==4));then for pid in "${pids[@]}";do wait "$pid"||failed=1;done;pids=();fi;done;done
for seed in 42 43 44 45 46 47;do arm=global_center_top3;run_one "$arm" "$seed" $((index%2))&pids+=("$!");index=$((index+1));if((${#pids[@]}==4));then for pid in "${pids[@]}";do wait "$pid"||failed=1;done;pids=();fi;done
for pid in "${pids[@]}";do wait "$pid"||failed=1;done;((failed==0))
python "$ROOT/CV-DD/fine_grained/summarize_dino_ipc3_hard_v1.py" --root "$EXP" --r0-results "$R0_RESULTS">"$EXP/logs/summary.log" 2>&1
rm -f "$EXP/status/running";date --iso-8601=seconds>"$EXP/status/complete"
