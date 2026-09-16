#!/usr/bin/env bash
set -euo pipefail
ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EXP="${EXP_ROOT:-/linxi/dataset/FGDD_compression/aircraft_ipc3_retarget_light_rrc_v1}"
STAGE="${STAGE_ROOT:-/tmp/fgdd_aircraft_ipc3_retarget_light_rrc_v1}"
R0=/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard/selected/A_imsize224/rseed0/ipc3
BBOX=/linxi/dataset/FGDD_compression/aircraft_ipc3_retarget_v1/construction/selected/bbox_retarget112/ipc3
TEACHER=/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42/ResNet18.pth
TEST=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/test
FULLFRAME_RESULTS=/linxi/dataset/FG_SoftAugFactorial_v2/aircraft_ipc3_v1/results/fullframe_cutmix/random_real/source_seed0
COMPRESSION=/linxi/dataset/FGDD_compression/aircraft_ipc3_retarget_v1
R0_FKD="$STAGE/fkd/r0/ipc3_bs20_ipc3"
BBOX_FKD="$STAGE/fkd/bbox_retarget112/ipc3_bs20_ipc3"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$EXP"/{logs,status,locks,fkd_archives,results,post_eval,audits,summary} "$STAGE/fkd"
exec 9>"$EXP/locks/launcher.lock"; flock -n 9 || exit 75
trap 's=$?; if ((s)); then rm -f "$EXP/status/running"; echo "$(date --iso-8601=seconds) exit=$s" > "$EXP/status/failed"; fi' EXIT
rm -f "$EXP/status/failed" "$EXP/status/complete"; date --iso-8601=seconds > "$EXP/status/running"

if [[ ! -f "$R0_FKD/relabel_manifest.json" && -f "$EXP/fkd_archives/r0.tar" ]]; then
 mkdir -p "$STAGE/fkd/r0"; tar -C "$STAGE/fkd/r0" -xf "$EXP/fkd_archives/r0.tar"
fi
if [[ ! -f "$R0_FKD/relabel_manifest.json" ]]; then
 CUDA_VISIBLE_DEVICES=1 python -u "$ROOT/CV-DD/relabel/relabel.py" --syn-data-path "$R0" \
  --fkd-path "$STAGE/fkd/r0/ipc3" --model-pool-dir "$(dirname "$TEACHER")" --teacher-model-name ResNet18 --gpu 0 \
  --batch-size 20 --workers 8 --persistent-workers --dataset-name A_imsize224 --epochs 400 --seed 42 --fkd-seed 42 \
  --min-scale-crops .8 --max-scale-crops 1 --min-aspect-ratio-crops 1 --max-aspect-ratio-crops 1 \
  --mix-type cutmix --use-fp16 --mode fkd_save > "$EXP/logs/relabel_r0.log" 2>&1
fi
python "$ROOT/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$R0_FKD" --images 300 --classes 100 --batch-size 20 \
 --epochs 400 --output "$EXP/audits/r0_fkd.json" >> "$EXP/logs/relabel_r0.log" 2>&1
if [[ ! -f "$EXP/fkd_archives/r0.tar" ]]; then
 tar -C "$STAGE/fkd/r0" -cf "$EXP/fkd_archives/r0.tar.tmp" ipc3_bs20_ipc3
 mv "$EXP/fkd_archives/r0.tar.tmp" "$EXP/fkd_archives/r0.tar"
fi

if [[ ! -f "$BBOX_FKD/relabel_manifest.json" && -f "$EXP/fkd_archives/bbox_retarget112.tar" ]]; then
 mkdir -p "$STAGE/fkd/bbox_retarget112"; tar -C "$STAGE/fkd/bbox_retarget112" -xf "$EXP/fkd_archives/bbox_retarget112.tar"
fi
if [[ ! -f "$BBOX_FKD/relabel_manifest.json" ]]; then
 CUDA_VISIBLE_DEVICES=1 python -u "$ROOT/CV-DD/fine_grained/rescore_fkd_views.py" --image-root "$BBOX" \
  --source-fkd "$R0_FKD" --output-fkd "$BBOX_FKD" --teacher "$TEACHER" --ipc 3 --classes 100 \
  --dataset-name A_imsize224 --epochs 400 --batch-size 20 --workers 8 --seed 42 --fkd-seed 42 \
  --mean .4865 .5177 .5425 --std .2124 .2051 .2375 > "$EXP/logs/relabel_bbox.log" 2>&1
fi
python "$ROOT/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$BBOX_FKD" --images 300 --classes 100 --batch-size 20 \
 --epochs 400 --output "$EXP/audits/bbox_fkd.json" >> "$EXP/logs/relabel_bbox.log" 2>&1
if [[ ! -f "$EXP/fkd_archives/bbox_retarget112.tar" ]]; then
 tar -C "$STAGE/fkd/bbox_retarget112" -cf "$EXP/fkd_archives/bbox_retarget112.tar.tmp" ipc3_bs20_ipc3
 mv "$EXP/fkd_archives/bbox_retarget112.tar.tmp" "$EXP/fkd_archives/bbox_retarget112.tar"
fi
python "$ROOT/CV-DD/fine_grained/audit_light_rrc_fkd.py" --reference "$R0_FKD" --candidate "$BBOX_FKD" \
 --output "$EXP/audits/light_rrc_trajectory.json" > "$EXP/logs/trajectory_audit.log" 2>&1
date --iso-8601=seconds > "$EXP/status/relabel.complete"

student(){
 local method=$1 seed=$2 gpu=$3 images fkd result
 if [[ "$method" == r0 ]]; then images="$R0"; fkd="$R0_FKD"; else images="$BBOX"; fkd="$BBOX_FKD"; fi
 result="$EXP/results/$method/ipc3_sseed${seed}.json"
 mkdir -p "$(dirname "$result")" "$EXP/post_eval/$method/sseed$seed"
 if [[ ! -f "$result" ]]; then
  CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 \
   --exp-name "retarget_light_rrc_${method}_s${seed}" --original-data-path "$images" --fkd-path "$fkd" \
   --output-dir "$EXP/post_eval/$method/sseed$seed" --batch-size 20 --epochs 400 --dataset-name A_imsize224 \
   --gradient-accumulation-steps 2 --mix-type cutmix --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$seed" \
   --temperature 20 --student-initialization imagenet-v1 --student-protocol-name "standard_v2_light_square_rrc_$method" \
   --adamw-weight-decay 1e-5 --adamw-beta1 .9 --adamw-beta2 .999 --adamw-eps 1e-8 --adamw-backbone-lr 1e-4 \
   --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 --val-dir "$TEST" --disable-wandb \
   --per-class-output "$result" > "$EXP/logs/eval_${method}_s${seed}.log" 2>&1
 fi
}
failed=0; pids=(); index=0
for method in r0 bbox_retarget112; do
 for seed in 42 43 44; do
  student "$method" "$seed" $((index % 2)) & pids+=("$!"); index=$((index + 1))
  if ((${#pids[@]} == 4)); then for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; pids=(); fi
 done
done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed == 0))
python "$ROOT/CV-DD/fine_grained/summarize_aircraft_retarget_light_rrc.py" --root "$EXP" \
 --fullframe-root "$FULLFRAME_RESULTS" --compression-root "$COMPRESSION" > "$EXP/logs/summary.log" 2>&1
rm -f "$EXP/status/running"; date --iso-8601=seconds > "$EXP/status/complete"
