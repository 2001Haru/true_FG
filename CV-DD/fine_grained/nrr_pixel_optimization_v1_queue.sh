#!/usr/bin/env bash
set -euo pipefail
[[ "${RUN_CONFIRMED:-0}" == 1 ]] || { echo "Refusing to run: set RUN_CONFIRMED=1 after protocol review" >&2; exit 64; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${EXP_ROOT:-/linxi/dataset/FGDD_NRR/aircraft_ipc3_pixel_optimization_v1}"
PROTOCOL="$ROOT/CV-DD/fine_grained/nrr_pixel_optimization_v1_protocol.json"
SELECTION=/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard/manifests/A_imsize224/rseed0/ipc3.json
TEACHER=/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42/ResNet18.pth
SOURCE_FKD=/linxi/dataset/FG_HardAugFactorial_v2/aircraft_ipc3_v1/fkd/fullframe_cutmix/random_real/source_seed0/ipc3_bs20_ipc3
R0_RESULTS=/linxi/dataset/FG_SoftAugFactorial_v2/aircraft_ipc3_v1/results/fullframe_cutmix/random_real/source_seed0
TEST=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/test
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_HOME=/linxi/models/torchvision_cache
mkdir -p "$EXP"/{construction,fkd,results,post_eval,logs,status,locks,audits,summary}
exec 9>"$EXP/locks/launcher.lock"; flock -n 9 || exit 75
trap 's=$?; if ((s)); then rm -f "$EXP/status/running"; echo "$(date --iso-8601=seconds) exit=$s" > "$EXP/status/failed"; fi' EXIT
rm -f "$EXP/status/failed" "$EXP/status/complete"; date --iso-8601=seconds > "$EXP/status/running"

if [[ ! -f "$EXP/construction/initial_manifest.json" ]]; then
 CUDA_VISIBLE_DEVICES=0 python -u "$ROOT/CV-DD/fine_grained/nrr_real_image_optimization.py" prepare \
  --protocol "$PROTOCOL" --output-root "$EXP/construction" --teacher "$TEACHER" --selection-manifest "$SELECTION" \
  --seed 42 > "$EXP/logs/construction_prepare.log" 2>&1
fi
python - "$EXP/construction/initial_manifest.json" "$PROTOCOL" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));p=json.load(open(sys.argv[2]))
assert x['status']=='complete' and x['images']==300 and x['classes']==100 and x['ipc']==3
assert x['protected_pixels_per_image']==15053 and abs(x['protected_fraction']-15053/50176)<1e-15
assert x['teacher_sha256']=='9a5759935df2b3e6db3c7555d8cc8efc4bb5b8a4b85b6fed2fac810387c06fe7'
assert p['status']=='frozen_pending_user_confirmation'
print('initialization and mask gate passed')
PY

optimize(){
 local arm=$1 gpu=$2
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/fine_grained/nrr_real_image_optimization.py" optimize \
  --protocol "$PROTOCOL" --output-root "$EXP/construction" --teacher "$TEACHER" --arm "$arm" --seed 42 \
  --updates 2000 --lr .01 --min-lr .0001 --bn-coefficient .01 --epsilon "$(python -c 'print(32/255)')" \
  > "$EXP/logs/construction_${arm}.log" 2>&1
}
failed=0
optimize plain 0 & p0=$!; optimize cam_protected 1 & p1=$!
wait "$p0" || failed=1; wait "$p1" || failed=1; ((failed==0))
optimize random_protected 0; 
python "$ROOT/CV-DD/fine_grained/nrr_real_image_optimization.py" finalize --protocol "$PROTOCOL" \
 --output-root "$EXP/construction" > "$EXP/logs/construction_finalize.log" 2>&1
date --iso-8601=seconds > "$EXP/status/construction.complete"

image_root(){ echo "$EXP/construction/selected/$1/ipc3"; }
relabel(){
 local arm=$1 gpu=$2
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/fine_grained/rescore_fkd_views.py" --image-root "$(image_root "$arm")" \
  --source-fkd "$SOURCE_FKD" --output-fkd "$EXP/fkd/$arm" --teacher "$TEACHER" --ipc 3 --classes 100 \
  --dataset-name A_imsize224 --epochs 400 --batch-size 20 --workers 8 --seed 42 --fkd-seed 42 \
  --mean .4865 .5177 .5425 --std .2124 .2051 .2375 > "$EXP/logs/relabel_${arm}.log" 2>&1
 python "$ROOT/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$EXP/fkd/$arm" --images 300 --classes 100 --batch-size 20 \
  --epochs 400 --output "$EXP/audits/${arm}_fkd.json" >> "$EXP/logs/relabel_${arm}.log" 2>&1
 python "$ROOT/CV-DD/fine_grained/audit_fkd_metadata_alignment.py" --reference "$SOURCE_FKD" --candidate "$EXP/fkd/$arm" \
  --output "$EXP/audits/${arm}_metadata.json" >> "$EXP/logs/relabel_${arm}.log" 2>&1
}
relabel plain 0 & p0=$!; relabel cam_protected 1 & p1=$!
wait "$p0" || failed=1; wait "$p1" || failed=1; ((failed==0))
relabel random_protected 0
date --iso-8601=seconds > "$EXP/status/relabel.complete"

student(){
 local arm=$1 seed=$2 gpu=$3 result="$EXP/results/$arm/ipc3_sseed$seed.json"
 mkdir -p "$(dirname "$result")" "$EXP/post_eval/$arm/sseed$seed"
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 \
  --exp-name "nrr_v1_${arm}_s${seed}" --original-data-path "$(image_root "$arm")" --fkd-path "$EXP/fkd/$arm" \
  --output-dir "$EXP/post_eval/$arm/sseed$seed" --batch-size 20 --epochs 400 --dataset-name A_imsize224 \
  --gradient-accumulation-steps 2 --mix-type cutmix --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$seed" \
  --temperature 20 --student-initialization imagenet-v1 --student-protocol-name "standard_v2_nrr_pixel_$arm" \
  --adamw-weight-decay 1e-5 --adamw-beta1 .9 --adamw-beta2 .999 --adamw-eps 1e-8 --adamw-backbone-lr 1e-4 \
  --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 --val-dir "$TEST" --disable-wandb \
  --per-class-output "$result" > "$EXP/logs/eval_${arm}_s${seed}.log" 2>&1
}
pids=(); index=0
for arm in plain cam_protected random_protected; do for seed in 42 43 44; do
 student "$arm" "$seed" $((index%2)) & pids+=("$!"); index=$((index+1))
 if ((${#pids[@]}==4)); then for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; pids=(); fi
done; done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; ((failed==0))
python "$ROOT/CV-DD/fine_grained/summarize_nrr_pixel_optimization_v1.py" --root "$EXP" --r0-results "$R0_RESULTS" \
 > "$EXP/logs/summary.log" 2>&1
rm -f "$EXP/status/running"; date --iso-8601=seconds > "$EXP/status/complete"
