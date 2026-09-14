#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOFT=/linxi/dataset/FGDD_global_local/aircraft_ipc3_seed0_v1
EXP="${EXP_ROOT:-/linxi/dataset/FGDD_global_local/aircraft_ipc3_seed0_hard_v1}"
RRC_FKD=/linxi/dataset/FG_RandomReal_soft_v2/aircraft_v1/fkd/A_imsize224/rseed0/ipc3_bs20_ipc3
TEST=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/test
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$EXP"/{logs,status,locks,results,post_eval,summary,audits}
exec 9>"$EXP/locks/launcher.lock"
flock -n 9 || exit 75
trap 's=$?; if ((s)); then rm -f "$EXP/status/running"; echo "$(date --iso-8601=seconds) exit=$s" > "$EXP/status/failed"; fi' EXIT
rm -f "$EXP/status/failed" "$EXP/status/complete"
date --iso-8601=seconds > "$EXP/status/running"

python - "$RRC_FKD/relabel_manifest.json" "$RRC_FKD/epoch_0/batch_0.tar" "$SOFT" <<'PY'
import json,sys,torch
from pathlib import Path
m=json.load(open(sys.argv[1]));payload=torch.load(sys.argv[2],map_location='cpu',weights_only=False);s=Path(sys.argv[3])
assert m['status']=='complete' and m['mix_type']=='cutmix'
full=torch.tensor([0.,0.,1.,1.])
assert any(not torch.allclose(torch.as_tensor(c).float(),full,rtol=0,atol=1e-7) for c in payload[0])
assert payload[2] is not None and payload[4] is not None
for method in ('lowres_detail','native_detail'):
 x=json.load(open(s/f'audits/{method}_metadata.json'));assert x['status']=='complete' and x['total_mismatches']==0
print('geometry gate passed')
PY

student(){
 local arm=$1 method=$2 seed=$3 gpu=$4 fkd
 if [[ $arm == fullframe_cutmix ]]; then fkd="$SOFT/fkd/$method"; else fkd="$RRC_FKD"; fi
 local images="$SOFT/construction/selected/$method/ipc3"
 mkdir -p "$EXP/results/$arm/$method" "$EXP/post_eval/$arm/$method/sseed$seed"
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 \
 --exp-name "global_local_hard_${arm}_${method}_s${seed}" --original-data-path "$images" --fkd-path "$fkd" \
 --output-dir "$EXP/post_eval/$arm/$method/sseed$seed" --batch-size 20 --epochs 400 --dataset-name A_imsize224 \
 --gradient-accumulation-steps 2 --fkd-hard-label --mix-type cutmix --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$seed" --temperature 20 \
 --student-initialization imagenet-v1 --student-protocol-name standard_v2_hard_global_local --adamw-weight-decay 1e-5 \
 --adamw-beta1 .9 --adamw-beta2 .999 --adamw-eps 1e-8 --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 \
 --cosine-t-max 400 --cosine-eta-min 0 --val-dir "$TEST" --disable-wandb --per-class-output "$EXP/results/$arm/$method/ipc3_sseed$seed.json" \
 > "$EXP/logs/${arm}_${method}_s${seed}.log" 2>&1
}

failed=0; pids=(); index=0
for arm in fullframe_cutmix rrc_cutmix; do
 for method in lowres_detail native_detail; do
  for seed in 42 43 44; do
   student "$arm" "$method" "$seed" $((index%2)) & pids+=("$!"); index=$((index+1))
   if ((${#pids[@]}==4)); then for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; pids=(); fi
  done
 done
done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0))
python "$ROOT/CV-DD/fine_grained/summarize_global_local_hard.py" --root "$EXP" --soft-root "$SOFT" > "$EXP/logs/summary.log" 2>&1
rm -f "$EXP/status/running"
date --iso-8601=seconds > "$EXP/status/complete"
