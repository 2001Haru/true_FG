#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SELECT=/linxi/dataset/FG_HardLabel_standard/v1/dino_ipc3_top3_kmeans3
WAIT_FOR=/linxi/dataset/FGDD_DeCO_style/aircraft_ipc3_seed_expansion_v1/status/complete
EXP="${EXP_ROOT:-/linxi/dataset/FG_DINO_selection_soft_v2/aircraft_ipc3_top3_kmeans3_v1}"
TEACHER=/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42/ResNet18.pth
TEST=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/test
FULL_FKD=/linxi/dataset/FG_HardAugFactorial_v2/aircraft_ipc3_v1/fkd/fullframe_cutmix/random_real/source_seed0/ipc3_bs20_ipc3
RRC_FKD=/linxi/dataset/FG_RandomReal_soft_v2/aircraft_v1/fkd/A_imsize224/rseed0/ipc3_bs20_ipc3
R0_FULL=/linxi/dataset/FG_SoftAugFactorial_v2/aircraft_ipc3_v1/results/fullframe_cutmix/random_real/source_seed0
R0_RRC=/linxi/dataset/FG_RandomReal_soft_v2/aircraft_v1/results/A_imsize224/rseed0
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_HOME=/linxi/models/torchvision_cache
WEIGHT_SOURCE=/linxi/models/torchvision/resnet18-f37072fd.pth;WEIGHT_CACHE="$TORCH_HOME/hub/checkpoints/resnet18-f37072fd.pth"
mkdir -p "$(dirname "$WEIGHT_CACHE")";[[ -e "$WEIGHT_CACHE" ]]||ln -s "$WEIGHT_SOURCE" "$WEIGHT_CACHE"
[[ "$(sha256sum "$WEIGHT_SOURCE"|cut -d' ' -f1)" == "$(sha256sum "$WEIGHT_CACHE"|cut -d' ' -f1)" ]]
mkdir -p "$EXP"/{selected,materialization_audits,logs,status,locks,fkd,results,post_eval,audits,summary}
exec 9>"$EXP/locks/launcher.lock";flock -n 9||exit 75
trap 's=$?;if((s));then rm -f "$EXP/status/running";echo "$(date --iso-8601=seconds) exit=$s">"$EXP/status/failed";fi' EXIT
rm -f "$EXP/status/failed" "$EXP/status/complete";date --iso-8601=seconds>"$EXP/status/waiting"
while [[ ! -f "$WAIT_FOR" ]];do sleep 30;done
rm -f "$EXP/status/waiting";date --iso-8601=seconds>"$EXP/status/running"

for arm in spherical_kmeans3_rseed0 spherical_kmeans3_rseed1 global_center_top3;do
 python "$ROOT/CV-DD/fine_grained/materialize_selection_manifest.py" --manifest "$SELECT/manifests/A_imsize224/$arm.json" \
  --output-root "$EXP/selected/$arm/ipc3" --audit-output "$EXP/materialization_audits/$arm.json" > "$EXP/logs/materialize_$arm.log" 2>&1
done
date --iso-8601=seconds>"$EXP/status/materialization.complete"
source_fkd(){ [[ "$1" == fullframe_cutmix ]]&&echo "$FULL_FKD"||echo "$RRC_FKD"; }
fkd(){ echo "$EXP/fkd/$1/$2"; }
fkd_complete(){ local p=$1 n=0 s='';[[ -d "$p" ]]&&n=$(find "$p" -type f -name 'batch_*.tar'|wc -l);[[ -f "$p/relabel_manifest.json" ]]&&s=$(python -c "import json;print(json.load(open('$p/relabel_manifest.json')).get('status',''))");[[ "$n" == 6000 && "$s" == complete ]]; }
relabel(){
 local mode=$1 arm=$2 gpu=$3 output;output="$(fkd "$mode" "$arm")"
 if ! fkd_complete "$output";then CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/fine_grained/rescore_fkd_views.py" \
  --image-root "$EXP/selected/$arm/ipc3" --source-fkd "$(source_fkd "$mode")" --output-fkd "$output" --teacher "$TEACHER" \
  --ipc 3 --classes 100 --dataset-name A_imsize224 --epochs 400 --batch-size 20 --workers 8 --seed 42 --fkd-seed 42 \
  --mean .4865 .5177 .5425 --std .2124 .2051 .2375 > "$EXP/logs/relabel_${mode}_${arm}.log" 2>&1;fi
 python "$ROOT/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$output" --images 300 --classes 100 --batch-size 20 --epochs 400 \
  --output "$EXP/audits/${mode}_${arm}_fkd.json" >> "$EXP/logs/relabel_${mode}_${arm}.log" 2>&1
 python "$ROOT/CV-DD/fine_grained/audit_fkd_metadata_alignment.py" --reference "$(source_fkd "$mode")" --candidate "$output" \
  --output "$EXP/audits/${mode}_${arm}_metadata.json" >> "$EXP/logs/relabel_${mode}_${arm}.log" 2>&1
}
failed=0;pids=();idx=0
for mode in fullframe_cutmix rrc_cutmix;do for arm in spherical_kmeans3_rseed0 spherical_kmeans3_rseed1 global_center_top3;do relabel "$mode" "$arm" $((idx%2))&pids+=("$!");idx=$((idx+1));if((${#pids[@]}==2));then for p in "${pids[@]}";do wait "$p"||failed=1;done;pids=();fi;done;done;((failed==0))
date --iso-8601=seconds>"$EXP/status/relabel.complete"

student(){
 local mode=$1 arm=$2 seed=$3 gpu=$4 result;result="$EXP/results/$mode/$arm/ipc3_sseed$seed.json"
 mkdir -p "$(dirname "$result")" "$EXP/post_eval/$mode/$arm/sseed$seed"
 if [[ ! -f "$result" ]];then CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 \
  --exp-name "dino_soft_${mode}_${arm}_s${seed}" --original-data-path "$EXP/selected/$arm/ipc3" --fkd-path "$(fkd "$mode" "$arm")" \
  --output-dir "$EXP/post_eval/$mode/$arm/sseed$seed" --batch-size 20 --epochs 400 --dataset-name A_imsize224 \
  --gradient-accumulation-steps 2 --mix-type cutmix --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$seed" \
  --temperature 20 --student-initialization imagenet-v1 --student-protocol-name "standard_v2_dino_ipc3_soft_$mode" \
  --adamw-weight-decay 1e-5 --adamw-beta1 .9 --adamw-beta2 .999 --adamw-eps 1e-8 --adamw-backbone-lr 1e-4 \
  --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 --val-dir "$TEST" --disable-wandb \
  --per-class-output "$result" > "$EXP/logs/eval_${mode}_${arm}_s${seed}.log" 2>&1;fi
}
pids=();idx=0
for mode in fullframe_cutmix rrc_cutmix;do for arm in spherical_kmeans3_rseed0 spherical_kmeans3_rseed1 global_center_top3;do for seed in 42 43 44;do student "$mode" "$arm" "$seed" $((idx%2))&pids+=("$!");idx=$((idx+1));if((${#pids[@]}==4));then for p in "${pids[@]}";do wait "$p"||failed=1;done;pids=();fi;done;done;done
for p in "${pids[@]}";do wait "$p"||failed=1;done;((failed==0))
python "$ROOT/CV-DD/fine_grained/summarize_dino_ipc3_soft_v2.py" --root "$EXP" --r0-full "$R0_FULL" --r0-rrc "$R0_RRC" > "$EXP/logs/summary.log" 2>&1
rm -f "$EXP/status/running";date --iso-8601=seconds>"$EXP/status/complete"
