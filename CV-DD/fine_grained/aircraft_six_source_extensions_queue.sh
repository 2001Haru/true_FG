#!/usr/bin/env bash
set -euo pipefail
ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EXP="${EXP_ROOT:-/tmp/fgdd_aircraft_six_source_extensions}"
PARENT=/tmp/fgdd_aircraft_six_source_experiment
PARENT_STAGE=/tmp/fgdd_aircraft_six_source_stage
RAW=/linxi/dataset/FD2/raw/fgvc-aircraft-2013b/data
TEACHER=/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42/ResNet18.pth
TEST=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/test
WEIGHTS=/linxi/models/torchvision/resnet18-f37072fd.pth
mkdir -p "$EXP"/{construction,logs,status,locks,fkd,results,checkpoints,post_eval,summary,audits}
exec 9>"$EXP/locks/launcher.lock"; flock -n 9 || exit 75
trap 's=$?; if ((s)); then rm -f "$EXP/status/running"; echo "$(date --iso-8601=seconds) exit=$s" > "$EXP/status/failed"; fi' EXIT
rm -f "$EXP/status/failed" "$EXP/status/complete"; date --iso-8601=seconds > "$EXP/status/running"

SEED0_MANIFEST="$EXP/construction/seed0_manifest.json"
if [[ ! -f "$SEED0_MANIFEST" ]]; then
 cp "$PARENT/construction/six_source_manifest.json" "$SEED0_MANIFEST"
 python "$ROOT/CV-DD/fine_grained/add_uniform_six_source_pack.py" --manifest "$SEED0_MANIFEST" \
  --output-root "$EXP/construction/uniform_seed0" > "$EXP/logs/uniform_construction.log" 2>&1
fi
for rseed in 1 2; do
 manifest="$EXP/construction/rseed$rseed/six_source_manifest.json"
 if [[ ! -f "$manifest" ]]; then
  python "$ROOT/CV-DD/fine_grained/prepare_aircraft_six_source_pack.py" \
   --selection-manifest "/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard/manifests/A_imsize224/rseed$rseed/ipc10.json" \
   --raw-images "$RAW/images" --boxes "$RAW/images_box.txt" --output-root "$EXP/construction/rseed$rseed" \
   > "$EXP/logs/construction_rseed$rseed.log" 2>&1
 fi
done
date --iso-8601=seconds > "$EXP/status/construction.complete"

UNIFORM_FKD="$EXP/fkd/uniform"
if [[ ! -f "$UNIFORM_FKD/relabel_manifest.json" ]]; then
 CUDA_VISIBLE_DEVICES=0 python -u "$ROOT/CV-DD/fine_grained/rescore_aircraft_six_source_mode.py" \
  --manifest "$SEED0_MANIFEST" --mode uniform --source-fkd "$PARENT_STAGE/fkd/reference" \
  --output-fkd "$UNIFORM_FKD" --teacher "$TEACHER" > "$EXP/logs/relabel_uniform.log" 2>&1
fi
python "$ROOT/CV-DD/fine_grained/audit_aircraft_six_source_fkd.py" --manifest "$SEED0_MANIFEST" \
 --reference "$PARENT_STAGE/fkd/reference" --compressed "$UNIFORM_FKD" \
 --output "$EXP/audits/uniform_fkd.json" > "$EXP/logs/audit_uniform.log" 2>&1
date --iso-8601=seconds > "$EXP/status/relabel.complete"

soft_uniform(){
 local seed=$1 gpu=$2 result="$EXP/results/soft_uniform/sseed$seed.json"
 mkdir -p "$(dirname "$result")" "$EXP/post_eval/soft_uniform/sseed$seed"
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 \
  --exp-name "six_source_soft_uniform_s$seed" --original-data-path "$EXP/construction/uniform_seed0" \
  --fkd-path "$UNIFORM_FKD" --paired-source-manifest "$SEED0_MANIFEST" --paired-source-mode uniform \
  --output-dir "$EXP/post_eval/soft_uniform/sseed$seed" --batch-size 20 --epochs 400 --dataset-name A_imsize224 \
  --gradient-accumulation-steps 2 --mix-type cutmix --workers 8 --persistent-workers --fkd_seed 42 \
  --train-seed "$seed" --temperature 20 --student-initialization imagenet-v1 \
  --student-protocol-name standard_v2_six_source_uniform_fullframe --adamw-weight-decay 1e-5 \
  --adamw-beta1 .9 --adamw-beta2 .999 --adamw-eps 1e-8 --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 \
  --cosine-t-max 400 --cosine-eta-min 0 --val-dir "$TEST" --disable-wandb --per-class-output "$result" \
  > "$EXP/logs/soft_uniform_s$seed.log" 2>&1
}

hard(){
 local mode=$1 rseed=$2 seed=$3 gpu=$4 manifest paired_mode suffix result
 if [[ "$mode" == uniform ]]; then
  manifest="$SEED0_MANIFEST"; paired_mode=uniform; suffix=uniform
 else
  manifest="$EXP/construction/rseed$rseed/six_source_manifest.json"; paired_mode=compressed; suffix="bbox_rseed$rseed"
 fi
 result="$EXP/results/hard_$suffix/sseed$seed.json"
 mkdir -p "$(dirname "$result")" "$EXP/checkpoints/$suffix/sseed$seed"
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_hard_label_v1.py" \
  --paired-source-manifest "$manifest" --paired-source-mode "$paired_mode" --val-dir "$TEST" \
  --dataset-name A_imsize224 --num-classes 100 --ipc 3 --student-seed "$seed" --result "$result" \
  --checkpoint-dir "$EXP/checkpoints/$suffix/sseed$seed" --imagenet-weights-path "$WEIGHTS" \
  --total-updates 3000 --batch-size 64 --backbone-lr 3e-4 --head-lr 3e-3 --backbone-min-lr 0 \
  --head-min-lr 0 --momentum .9 --weight-decay 5e-4 --eval-every-updates 300 --workers 8 \
  --persistent-workers --val-batch-size 256 --protocol-name "hard_label_v1_six_source_mild_rc_$suffix" \
  --train-crop-mode mild_rc > "$EXP/logs/hard_${suffix}_s$seed.log" 2>&1
}

failed=0; pids=(); index=0
for seed in 42 43 44; do
 soft_uniform "$seed" $((index%2)) & pids+=("$!"); index=$((index+1))
 hard uniform 0 "$seed" $((index%2)) & pids+=("$!"); index=$((index+1))
 if ((${#pids[@]}==4)); then for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; pids=(); ((failed==0)); fi
done
for rseed in 1 2; do
 for seed in 42 43 44; do
  hard bbox "$rseed" "$seed" $((index%2)) & pids+=("$!"); index=$((index+1))
  if ((${#pids[@]}==4)); then for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; pids=(); ((failed==0)); fi
 done
done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed == 0))
python "$ROOT/CV-DD/fine_grained/summarize_aircraft_six_source_extensions.py" --root "$EXP" --parent "$PARENT" \
 > "$EXP/logs/summary.log" 2>&1
rm -f "$EXP/status/running"; date --iso-8601=seconds > "$EXP/status/complete"
