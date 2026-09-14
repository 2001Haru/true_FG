#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${EXP_ROOT:-/linxi/dataset/FGDD_DeCO_style/aircraft_ipc3_seed0_v1}"
RANDOM_REAL_ROOT=/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard
R0="$RANDOM_REAL_ROOT/selected/A_imsize224/rseed0/ipc3"
SELECTION="$RANDOM_REAL_ROOT/manifests/A_imsize224/rseed0/ipc3.json"
TRAIN=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/train
RAW=/linxi/dataset/FD2/raw/fgvc-aircraft-2013b/data
TRANS=/linxi/dataset/FG_ViT_Teachers/aircraft_seed42_224_v1
TEACHER=/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42/ResNet18.pth
SOURCE_FKD=/linxi/dataset/FG_HardAugFactorial_v2/aircraft_ipc3_v1/fkd/fullframe_cutmix/random_real/source_seed0/ipc3_bs20_ipc3
R0_RESULTS=/linxi/dataset/FG_SoftAugFactorial_v2/aircraft_ipc3_v1/results/fullframe_cutmix/random_real/source_seed0
TEST=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/test
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$EXP"/{construction,logs,status,locks,fkd,results,post_eval,audits,summary}
exec 9>"$EXP/locks/launcher.lock"
flock -n 9 || exit 75
trap 's=$?; if ((s)); then rm -f "$EXP/status/running"; echo "$(date --iso-8601=seconds) exit=$s" > "$EXP/status/failed"; fi' EXIT
rm -f "$EXP/status/failed" "$EXP/status/complete"
date --iso-8601=seconds > "$EXP/status/running"

if [[ ! -f "$EXP/construction/construction_manifest.json" ]]; then
 CUDA_VISIBLE_DEVICES=0 python -u "$ROOT/CV-DD/fine_grained/prepare_deco_style_aircraft.py" \
  --selection-manifest "$SELECTION" --train-root "$TRAIN" --raw-images "$RAW/images" --boxes "$RAW/images_box.txt" \
  --transfg-source "$ROOT/third_party/teacher_backbones/TransFG" --transfg-checkpoint "$TRANS/teachers/transfg/final_step10000.pth" \
  --output-root "$EXP/construction" > "$EXP/logs/construction.log" 2>&1
fi
python - "$EXP/construction/construction_manifest.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));assert x['status']=='complete' and x['stored_images']==300 and x['regions']==1200
assert x['independent_sources_per_class']==12 and x['r0_sources_per_class']==3 and x['additional_sources_per_class']==9
assert all(v['images']==300 for v in x['outputs'].values()) and x['fg_attention_higher_fraction']>.95
for c in range(100):
 rows=[r for r in x['regions_detail'] if r['class_id']==c];assert len(rows)==12 and len({r['image_id'] for r in rows})==12 and sum(r['is_r0_source'] for r in rows)==3
print('construction gate passed')
PY
date --iso-8601=seconds > "$EXP/status/construction.complete"

relabel(){
 local method=$1 gpu=$2
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/fine_grained/rescore_fkd_views.py" --image-root "$EXP/construction/selected/$method/ipc3" \
 --source-fkd "$SOURCE_FKD" --output-fkd "$EXP/fkd/$method" --teacher "$TEACHER" --ipc 3 --classes 100 --dataset-name A_imsize224 \
 --epochs 400 --batch-size 20 --workers 8 --seed 42 --fkd-seed 42 --mean .4865 .5177 .5425 --std .2124 .2051 .2375 > "$EXP/logs/relabel_$method.log" 2>&1
 python "$ROOT/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$EXP/fkd/$method" --images 300 --classes 100 --batch-size 20 --epochs 400 --output "$EXP/audits/${method}_fkd.json" >> "$EXP/logs/relabel_$method.log" 2>&1
 python "$ROOT/CV-DD/fine_grained/audit_fkd_metadata_alignment.py" --reference "$SOURCE_FKD" --candidate "$EXP/fkd/$method" --output "$EXP/audits/${method}_metadata.json" >> "$EXP/logs/relabel_$method.log" 2>&1
}
relabel random_regions 0 & p0=$!
relabel fg_regions 1 & p1=$!
failed=0;wait "$p0"||failed=1;wait "$p1"||failed=1;((failed==0))
date --iso-8601=seconds > "$EXP/status/relabel.complete"

student(){
 local method=$1 seed=$2 gpu=$3
 mkdir -p "$EXP/results/$method" "$EXP/post_eval/$method/sseed$seed"
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 --exp-name "deco_${method}_s${seed}" \
 --original-data-path "$EXP/construction/selected/$method/ipc3" --fkd-path "$EXP/fkd/$method" --output-dir "$EXP/post_eval/$method/sseed$seed" \
 --batch-size 20 --epochs 400 --dataset-name A_imsize224 --gradient-accumulation-steps 2 --mix-type cutmix --workers 8 --persistent-workers \
 --fkd_seed 42 --train-seed "$seed" --temperature 20 --student-initialization imagenet-v1 --student-protocol-name "standard_v2_deco_$method" \
 --adamw-weight-decay 1e-5 --adamw-beta1 .9 --adamw-beta2 .999 --adamw-eps 1e-8 --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 \
 --cosine-t-max 400 --cosine-eta-min 0 --val-dir "$TEST" --disable-wandb --per-class-output "$EXP/results/$method/ipc3_sseed$seed.json" > "$EXP/logs/eval_${method}_s${seed}.log" 2>&1
}
pids=();index=0
for method in random_regions fg_regions;do for seed in 42 43 44;do student "$method" "$seed" $((index%2))&pids+=("$!");index=$((index+1));if((${#pids[@]}==4));then for pid in "${pids[@]}";do wait "$pid"||failed=1;done;pids=();fi;done;done
for pid in "${pids[@]}";do wait "$pid"||failed=1;done;((failed==0))
python "$ROOT/CV-DD/fine_grained/summarize_deco_style_aircraft.py" --root "$EXP" --r0-results "$R0_RESULTS" > "$EXP/logs/summary.log" 2>&1
rm -f "$EXP/status/running";date --iso-8601=seconds > "$EXP/status/complete"
