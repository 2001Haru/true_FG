#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${EXP_ROOT:-/linxi/dataset/FGDD_compression/aircraft_ipc3_retarget_v1}"
R0=/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard/selected/A_imsize224/rseed0/ipc3
RAW=/linxi/dataset/FD2/raw/fgvc-aircraft-2013b/data
TRANS=/linxi/dataset/FG_ViT_Teachers/aircraft_seed42_224_v1
TEACHER=/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42/ResNet18.pth
SOURCE_FKD=/linxi/dataset/FG_HardAugFactorial_v2/aircraft_ipc3_v1/fkd/fullframe_cutmix/random_real/source_seed0/ipc3_bs20_ipc3
R0_RESULTS=/linxi/dataset/FG_SoftAugFactorial_v2/aircraft_ipc3_v1/results/fullframe_cutmix/random_real/source_seed0
TEST=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/test
METHODS=(isotropic158 vertical112 horizontal112 bbox_retarget112 attention_retarget112)
SEEDS=(42 43 44)
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$EXP"/{construction,logs,status,locks,fkd,results,post_eval,audits,summary}
exec 9>"$EXP/locks/launcher.lock"; flock -n 9 || exit 75
trap 's=$?; if ((s)); then rm -f "$EXP/status/running"; echo "$(date --iso-8601=seconds) exit=$s" > "$EXP/status/failed"; fi' EXIT
rm -f "$EXP/status/failed" "$EXP/status/complete"; date --iso-8601=seconds > "$EXP/status/running"

if [[ ! -f "$EXP/construction/construction_manifest.json" ]]; then
 CUDA_VISIBLE_DEVICES=0 python -u "$ROOT/CV-DD/fine_grained/prepare_aircraft_retarget_compression.py" \
  --r0-root "$R0" --raw-images "$RAW/images" --boxes "$RAW/images_box.txt" \
  --transfg-source "$ROOT/third_party/teacher_backbones/TransFG" \
  --transfg-checkpoint "$TRANS/teachers/transfg/final_step10000.pth" \
  --output-root "$EXP/construction" > "$EXP/logs/construction.log" 2>&1
fi
python - "$EXP/construction/construction_manifest.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['status']=='complete' and x['images']==300 and x['classes']==100
assert x['isotropic_pixel_budget']==158**2 and x['strip_pixel_budget']==224*112
assert abs(x['isotropic_minus_strip_fraction'] - (158**2-224*112)/(224*112)) < 1e-12
assert set(x['outputs'])=={'isotropic158','vertical112','horizontal112','bbox_retarget112','attention_retarget112'}
assert all(v['images']==300 for v in x['outputs'].values())
b=x['bbox_audit']; assert b['subject_density']['max']<=.9000001 and b['background_density']['min']>=.0999999
a=x['attention_density_audit']; assert a['minimum']['min']>=.0999999 and a['maximum']['max']<=.9000001
print('construction gate passed', {'bbox_reduced':b['reduced_images'],'bbox_height':b['height224'],'attention_min':a['minimum'],'attention_max':a['maximum']})
PY
date --iso-8601=seconds > "$EXP/status/construction.complete"

relabel(){
 local method=$1 gpu=$2 out="$EXP/fkd/$method"
 if [[ ! -f "$out/relabel_manifest.json" ]]; then
  CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/fine_grained/rescore_fkd_views.py" \
   --image-root "$EXP/construction/selected/$method/ipc3" --source-fkd "$SOURCE_FKD" --output-fkd "$out" \
   --teacher "$TEACHER" --ipc 3 --classes 100 --dataset-name A_imsize224 --epochs 400 --batch-size 20 \
   --workers 8 --seed 42 --fkd-seed 42 --mean .4865 .5177 .5425 --std .2124 .2051 .2375 \
   > "$EXP/logs/relabel_$method.log" 2>&1
 fi
 python "$ROOT/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$out" --images 300 --classes 100 --batch-size 20 \
  --epochs 400 --output "$EXP/audits/${method}_fkd.json" >> "$EXP/logs/relabel_$method.log" 2>&1
 python "$ROOT/CV-DD/fine_grained/audit_fkd_metadata_alignment.py" --reference "$SOURCE_FKD" --candidate "$out" \
  --output "$EXP/audits/${method}_metadata.json" >> "$EXP/logs/relabel_$method.log" 2>&1
}
failed=0; pids=(); index=0
for method in "${METHODS[@]}"; do
 relabel "$method" $((index % 2)) & pids+=("$!"); index=$((index + 1))
 if ((${#pids[@]} == 2)); then for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; pids=(); fi
done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed == 0)); date --iso-8601=seconds > "$EXP/status/relabel.complete"

student(){
 local method=$1 seed=$2 gpu=$3 result="$EXP/results/$method/ipc3_sseed${seed}.json"
 mkdir -p "$(dirname "$result")" "$EXP/post_eval/$method/sseed$seed"
 if [[ ! -f "$result" ]]; then
  CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 \
   --exp-name "retarget_${method}_s${seed}" --original-data-path "$EXP/construction/selected/$method/ipc3" \
   --fkd-path "$EXP/fkd/$method" --output-dir "$EXP/post_eval/$method/sseed$seed" --batch-size 20 --epochs 400 \
   --dataset-name A_imsize224 --gradient-accumulation-steps 2 --mix-type cutmix --workers 8 --persistent-workers \
   --fkd_seed 42 --train-seed "$seed" --temperature 20 --student-initialization imagenet-v1 \
   --student-protocol-name "standard_v2_retarget_$method" --adamw-weight-decay 1e-5 --adamw-beta1 .9 \
   --adamw-beta2 .999 --adamw-eps 1e-8 --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 \
   --cosine-t-max 400 --cosine-eta-min 0 --val-dir "$TEST" --disable-wandb --per-class-output "$result" \
   > "$EXP/logs/eval_${method}_s${seed}.log" 2>&1
 fi
}
pids=(); index=0
for method in "${METHODS[@]}"; do
 for seed in "${SEEDS[@]}"; do
  student "$method" "$seed" $((index % 2)) & pids+=("$!"); index=$((index + 1))
  if ((${#pids[@]} == 4)); then for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; pids=(); fi
 done
done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed == 0))
python "$ROOT/CV-DD/fine_grained/summarize_aircraft_retarget_compression.py" --root "$EXP" --r0-results "$R0_RESULTS" \
 > "$EXP/logs/summary.log" 2>&1
rm -f "$EXP/status/running"; date --iso-8601=seconds > "$EXP/status/complete"
