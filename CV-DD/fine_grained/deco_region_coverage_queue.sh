#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${EXP_ROOT:-/linxi/dataset/FGDD_DeCO_style/aircraft_ipc3_region_coverage_seed0_v1}"
BASE=/linxi/dataset/FGDD_DeCO_style/aircraft_ipc3_seed0_v1/construction/construction_manifest.json
TRANS=/linxi/dataset/FG_ViT_Teachers/aircraft_seed42_224_v1
DINO=/linxi/models/DINOv2/dinov2-base
SOURCE_FKD=/linxi/dataset/FG_RandomReal_soft_v2/aircraft_v1/fkd/A_imsize224/rseed0/ipc3_bs20_ipc3
TEACHER=/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42/ResNet18.pth
TEST=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/test
WEIGHTS=/linxi/models/torchvision/resnet18-f37072fd.pth
SOFT_FG=/linxi/dataset/FGDD_DeCO_style/aircraft_ipc3_seed0_rrc_v1/results/fg_regions
HARD_FG=/linxi/dataset/FG_HardLabel_standard/v1/aircraft_ipc3_rc_rrc_regions_v1/results/rrc/fg_regions
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_HOME=/linxi/models/torchvision_cache
mkdir -p "$TORCH_HOME/hub/checkpoints" "$EXP"/{construction,logs,status,locks,fkd,results,post_eval,checkpoints,audits,summary}
[[ -e "$TORCH_HOME/hub/checkpoints/resnet18-f37072fd.pth" ]] || ln -s "$WEIGHTS" "$TORCH_HOME/hub/checkpoints/resnet18-f37072fd.pth"
exec 9>"$EXP/locks/launcher.lock"; flock -n 9 || exit 75
trap 's=$?; if ((s)); then rm -f "$EXP/status/running"; echo "$(date --iso-8601=seconds) exit=$s" > "$EXP/status/failed"; fi' EXIT
rm -f "$EXP/status/failed" "$EXP/status/complete"; date --iso-8601=seconds > "$EXP/status/running"

if [[ ! -f "$EXP/construction/construction_manifest.json" ]]; then
 CUDA_VISIBLE_DEVICES=0 python -u "$ROOT/CV-DD/fine_grained/prepare_deco_region_coverage_aircraft.py" \
  --base-manifest "$BASE" --transfg-source "$ROOT/third_party/teacher_backbones/TransFG" \
  --transfg-checkpoint "$TRANS/teachers/transfg/final_step10000.pth" --dino-model-root "$DINO" \
  --output-root "$EXP/construction" --selection-seed 20260915 > "$EXP/logs/construction.log" 2>&1
fi
python - "$EXP/construction/construction_manifest.json" "$BASE" <<'PY'
import hashlib,json,sys
from pathlib import Path
x=json.loads(Path(sys.argv[1]).read_text()); b=json.loads(Path(sys.argv[2]).read_text())
assert x['status']=='complete' and x['stored_images']==300 and x['regions']==1200
assert x['independent_sources_per_class']==12 and x['candidates_per_source']==4
assert all(v['images']==300 for v in x['outputs'].values())
assert len(x['regions_detail'])==len(b['regions_detail'])==1200
for new,old in zip(x['regions_detail'],b['regions_detail']):
 assert (new['class'],new['mosaic'],new['tile'],new['image_id'],new['raw_path'])==(old['class'],old['mosaic'],old['tile'],old['image_id'],old['raw_path'])
 assert new['candidates'][0]['window224']==old['fg_window224']
 assert 1<=len(new['candidates'])<=4 and len({tuple(c['window224']) for c in new['candidates']})==len(new['candidates'])
for row in x['class_audits']:
 assert row['joint_coverage_utility']+1e-8>=row['current_fg_utility']
 assert row['joint_coverage_utility']+1e-8>=row['candidate_random_utility']
print('construction gate passed')
PY
date --iso-8601=seconds > "$EXP/status/construction.complete"

image_root(){ echo "$EXP/construction/selected/$1/ipc3"; }
relabel(){
 local method=$1 gpu=$2
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/fine_grained/rescore_fkd_views.py" --image-root "$(image_root "$method")" \
  --source-fkd "$SOURCE_FKD" --output-fkd "$EXP/fkd/$method" --teacher "$TEACHER" --ipc 3 --classes 100 \
  --dataset-name A_imsize224 --epochs 400 --batch-size 20 --workers 8 --seed 42 --fkd-seed 42 \
  --mean .4865 .5177 .5425 --std .2124 .2051 .2375 > "$EXP/logs/relabel_$method.log" 2>&1
 python "$ROOT/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$EXP/fkd/$method" --images 300 --classes 100 \
  --batch-size 20 --epochs 400 --output "$EXP/audits/${method}_fkd.json" >> "$EXP/logs/relabel_$method.log" 2>&1
 python "$ROOT/CV-DD/fine_grained/audit_fkd_metadata_alignment.py" --reference "$SOURCE_FKD" --candidate "$EXP/fkd/$method" \
  --output "$EXP/audits/${method}_metadata.json" >> "$EXP/logs/relabel_$method.log" 2>&1
}
failed=0
relabel candidate_random 0 & p0=$!; relabel joint_coverage 1 & p1=$!
wait "$p0" || failed=1; wait "$p1" || failed=1; ((failed==0))
date --iso-8601=seconds > "$EXP/status/relabel.complete"

soft_student(){
 local method=$1 seed=$2 gpu=$3 result="$EXP/results/soft/$method/ipc3_sseed$seed.json"
 mkdir -p "$(dirname "$result")" "$EXP/post_eval/soft/$method/sseed$seed"
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 \
  --exp-name "deco_coverage_soft_${method}_s${seed}" --original-data-path "$(image_root "$method")" --fkd-path "$EXP/fkd/$method" \
  --output-dir "$EXP/post_eval/soft/$method/sseed$seed" --batch-size 20 --epochs 400 --dataset-name A_imsize224 \
  --gradient-accumulation-steps 2 --mix-type cutmix --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$seed" \
  --temperature 20 --student-initialization imagenet-v1 --student-protocol-name "standard_v2_deco_coverage_$method" \
  --adamw-weight-decay 1e-5 --adamw-beta1 .9 --adamw-beta2 .999 --adamw-eps 1e-8 --adamw-backbone-lr 1e-4 \
  --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 --val-dir "$TEST" --disable-wandb --per-class-output "$result" \
  > "$EXP/logs/eval_soft_${method}_s${seed}.log" 2>&1
}
hard_student(){
 local method=$1 seed=$2 gpu=$3 result="$EXP/results/hard_v1_rrc/$method/sseed$seed.json"
 mkdir -p "$(dirname "$result")" "$EXP/checkpoints/hard_v1_rrc/$method/sseed$seed"
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_hard_label_v1.py" --train-dir "$(image_root "$method")" \
  --val-dir "$TEST" --dataset-name A_imsize224 --num-classes 100 --ipc 3 --student-seed "$seed" --result "$result" \
  --checkpoint-dir "$EXP/checkpoints/hard_v1_rrc/$method/sseed$seed" --imagenet-weights-path "$WEIGHTS" --total-updates 3000 \
  --batch-size 64 --backbone-lr 3e-4 --head-lr 3e-3 --backbone-min-lr 0 --head-min-lr 0 --momentum .9 \
  --weight-decay 5e-4 --eval-every-updates 300 --workers 8 --persistent-workers --val-batch-size 256 \
  --protocol-name hard_label_v1_rrc --train-crop-mode rrc --rrc-min-scale .08 --rrc-max-scale 1 \
  > "$EXP/logs/eval_hard_v1_rrc_${method}_s${seed}.log" 2>&1
 python "$ROOT/CV-DD/fine_grained/audit_hard_label_v1_result.py" --result "$result" --dataset A_imsize224 --classes 100 \
  --ipc 3 --student-seed "$seed" --validation-images 3333 --protocol-name hard_label_v1_rrc --train-crop-mode rrc \
  --rrc-min-scale .08 --rrc-max-scale 1 >> "$EXP/logs/eval_hard_v1_rrc_${method}_s${seed}.log" 2>&1
}
pids=(); index=0
for protocol in soft hard; do for method in candidate_random joint_coverage; do for seed in 42 43 44; do
 if [[ $protocol == soft ]]; then soft_student "$method" "$seed" $((index%2)) & else hard_student "$method" "$seed" $((index%2)) & fi
 pids+=("$!"); index=$((index+1))
 if ((${#pids[@]}==4)); then for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; pids=(); fi
done; done; done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; ((failed==0))
python "$ROOT/CV-DD/fine_grained/summarize_deco_region_coverage_aircraft.py" --root "$EXP" \
 --soft-fg-reference "$SOFT_FG" --hard-fg-reference "$HARD_FG" > "$EXP/logs/summary.log" 2>&1
rm -f "$EXP/status/running"; date --iso-8601=seconds > "$EXP/status/complete"
