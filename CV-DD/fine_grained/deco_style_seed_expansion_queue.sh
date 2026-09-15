#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${EXP_ROOT:-/linxi/dataset/FGDD_DeCO_style/aircraft_ipc3_seed_expansion_v1}"
RANDOM_ROOT=/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard
R0_SOFT=/linxi/dataset/FG_RandomReal_soft_v2/aircraft_v1
TRAIN=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/train
RAW=/linxi/dataset/FD2/raw/fgvc-aircraft-2013b/data
TRANS=/linxi/dataset/FG_ViT_Teachers/aircraft_seed42_224_v1
TEACHER=/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42/ResNet18.pth
TEST=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/test
SEED0=/linxi/dataset/FGDD_DeCO_style/aircraft_ipc3_seed0_rrc_v1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$EXP"/{construction,logs,status,locks,fkd,results,post_eval,audits,summary}
exec 9>"$EXP/locks/launcher.lock";flock -n 9||exit 75
trap 's=$?;if((s));then rm -f "$EXP/status/running";echo "$(date --iso-8601=seconds) exit=$s">"$EXP/status/failed";fi' EXIT
rm -f "$EXP/status/failed" "$EXP/status/complete";date --iso-8601=seconds>"$EXP/status/running"

construct(){
 local cseed=$1 stable_seed=$2 gpu=$3 out
 out="$EXP/construction/cseed$cseed"
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/fine_grained/prepare_deco_style_aircraft.py" \
  --selection-manifest "$RANDOM_ROOT/manifests/A_imsize224/rseed$cseed/ipc3.json" --train-root "$TRAIN" \
  --raw-images "$RAW/images" --boxes "$RAW/images_box.txt" --transfg-source "$ROOT/third_party/teacher_backbones/TransFG" \
  --transfg-checkpoint "$TRANS/teachers/transfg/final_step10000.pth" --output-root "$out" \
  --selection-seed "$stable_seed" > "$EXP/logs/construction_cseed$cseed.log" 2>&1
}
construct 1 20260915 0&p0=$!;construct 2 20260916 1&p1=$!;failed=0;wait "$p0"||failed=1;wait "$p1"||failed=1;((failed==0))
python - "$EXP" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for cseed,stable in ((1,20260915),(2,20260916)):
 x=json.loads((root/f'construction/cseed{cseed}/construction_manifest.json').read_text())
 assert x['status']=='complete' and x['selection_seed']==stable and x['r0_selection_seed']==cseed
 assert x['stored_images']==300 and x['regions']==1200 and x['independent_sources_per_class']==12
 assert x['fg_attention_higher_fraction']>.95
 for c in range(100):
  rows=[r for r in x['regions_detail'] if r['class_id']==c]
  assert len(rows)==12 and len({r['image_id'] for r in rows})==12 and sum(r['is_r0_source'] for r in rows)==3
print('construction gate passed')
PY
date --iso-8601=seconds>"$EXP/status/construction.complete"

source_fkd(){ echo "$R0_SOFT/fkd/A_imsize224/rseed$1/ipc3_bs20_ipc3"; }
image_root(){ echo "$EXP/construction/cseed$1/selected/$2/ipc3"; }
fkd(){ echo "$EXP/fkd/cseed$1/$2"; }
relabel(){
 local cseed=$1 method=$2 gpu=$3
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/fine_grained/rescore_fkd_views.py" --image-root "$(image_root "$cseed" "$method")" \
  --source-fkd "$(source_fkd "$cseed")" --output-fkd "$(fkd "$cseed" "$method")" --teacher "$TEACHER" \
  --ipc 3 --classes 100 --dataset-name A_imsize224 --epochs 400 --batch-size 20 --workers 8 --seed 42 --fkd-seed 42 \
  --mean .4865 .5177 .5425 --std .2124 .2051 .2375 > "$EXP/logs/relabel_cseed${cseed}_${method}.log" 2>&1
 python "$ROOT/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$(fkd "$cseed" "$method")" --images 300 --classes 100 \
  --batch-size 20 --epochs 400 --output "$EXP/audits/cseed${cseed}_${method}_fkd.json" >> "$EXP/logs/relabel_cseed${cseed}_${method}.log" 2>&1
 python "$ROOT/CV-DD/fine_grained/audit_fkd_metadata_alignment.py" --reference "$(source_fkd "$cseed")" \
  --candidate "$(fkd "$cseed" "$method")" --output "$EXP/audits/cseed${cseed}_${method}_metadata.json" >> "$EXP/logs/relabel_cseed${cseed}_${method}.log" 2>&1
}
pids=();idx=0
for cseed in 1 2;do for method in random_regions fg_regions;do relabel "$cseed" "$method" $((idx%2))&pids+=("$!");idx=$((idx+1));if((${#pids[@]}==2));then for p in "${pids[@]}";do wait "$p"||failed=1;done;pids=();fi;done;done;((failed==0))
date --iso-8601=seconds>"$EXP/status/relabel.complete"

student(){
 local cseed=$1 method=$2 seed=$3 gpu=$4 result
 result="$EXP/results/cseed$cseed/$method/ipc3_sseed$seed.json";mkdir -p "$(dirname "$result")" "$EXP/post_eval/cseed$cseed/$method/sseed$seed"
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 \
  --exp-name "deco_cseed${cseed}_${method}_s${seed}" --original-data-path "$(image_root "$cseed" "$method")" \
  --fkd-path "$(fkd "$cseed" "$method")" --output-dir "$EXP/post_eval/cseed$cseed/$method/sseed$seed" \
  --batch-size 20 --epochs 400 --dataset-name A_imsize224 --gradient-accumulation-steps 2 --mix-type cutmix \
  --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$seed" --temperature 20 --student-initialization imagenet-v1 \
  --student-protocol-name "standard_v2_deco_cseed${cseed}_$method" --adamw-weight-decay 1e-5 --adamw-beta1 .9 \
  --adamw-beta2 .999 --adamw-eps 1e-8 --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 \
  --cosine-t-max 400 --cosine-eta-min 0 --val-dir "$TEST" --disable-wandb --per-class-output "$result" \
  > "$EXP/logs/eval_cseed${cseed}_${method}_s${seed}.log" 2>&1
}
pids=();idx=0
for cseed in 1 2;do for method in random_regions fg_regions;do for seed in 42 43 44;do student "$cseed" "$method" "$seed" $((idx%2))&pids+=("$!");idx=$((idx+1));if((${#pids[@]}==4));then for p in "${pids[@]}";do wait "$p"||failed=1;done;pids=();fi;done;done;done
for p in "${pids[@]}";do wait "$p"||failed=1;done;((failed==0))
python "$ROOT/CV-DD/fine_grained/summarize_deco_style_seed_expansion.py" --root "$EXP" --seed0-root "$SEED0" \
 --r0-results-root "$R0_SOFT/results/A_imsize224" > "$EXP/logs/summary.log" 2>&1
rm -f "$EXP/status/running";date --iso-8601=seconds>"$EXP/status/complete"
