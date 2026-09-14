#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_EXP=/linxi/dataset/FGDD_DeCO_style/aircraft_ipc3_seed0_v1
EXP="${EXP_ROOT:-/linxi/dataset/FGDD_DeCO_style/aircraft_ipc3_seed0_rrc_v1}"
CONSTRUCTION="$SOURCE_EXP/construction/construction_manifest.json"
SOURCE_FKD=/linxi/dataset/FG_RandomReal_soft_v2/aircraft_v1/fkd/A_imsize224/rseed0/ipc3_bs20_ipc3
R0_RESULTS=/linxi/dataset/FG_RandomReal_soft_v2/aircraft_v1/results/A_imsize224/rseed0
TEACHER=/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42/ResNet18.pth
TEST=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/test
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$EXP"/{logs,status,locks,fkd,results,post_eval,audits,summary}
exec 9>"$EXP/locks/launcher.lock"
flock -n 9 || exit 75
trap 's=$?; if ((s)); then rm -f "$EXP/status/running"; echo "$(date --iso-8601=seconds) exit=$s" > "$EXP/status/failed"; fi' EXIT
rm -f "$EXP/status/failed" "$EXP/status/complete"
date --iso-8601=seconds > "$EXP/status/running"

python - "$CONSTRUCTION" "$SOURCE_FKD/relabel_manifest.json" "$EXP/experiment_definition.json" "$(git -C "$ROOT" rev-parse HEAD)" <<'PY'
import json, os, sys
from pathlib import Path
construction, source_manifest, output = map(Path, sys.argv[1:4])
revision = sys.argv[4]
c=json.loads(construction.read_text()); s=json.loads(source_manifest.read_text())
assert c['status']=='complete' and c['stored_images']==300 and c['regions']==1200
assert s['status']=='complete' and s['mix_type']=='cutmix'
x={'status':'running','protocol':'controlled_deco_style_regions_rrc_cutmix_v1','git_revision':revision,
   'construction_manifest':str(construction),'source_fkd':str(source_manifest.parent),
   'view':'RRC scale [0.08,1] + flip + CutMix','methods':['random_regions','fg_regions'],
   'student_seeds':[42,43,44],'new_fkd_sets':2,'new_student_runs':6,
   'invariants':'same stored mosaics, source/layout, Teacher42 BSSL, FKD seed42, Student v2; only enable RRC relative to prior DeCO-style run'}
output.parent.mkdir(parents=True,exist_ok=True); tmp=output.with_suffix('.json.tmp'); tmp.write_text(json.dumps(x,indent=2)+'\n'); os.replace(tmp,output)
PY

relabel(){
 local method=$1 gpu=$2 image_root
 image_root="$SOURCE_EXP/construction/selected/$method/ipc3"
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/fine_grained/rescore_fkd_views.py" --image-root "$image_root" \
  --source-fkd "$SOURCE_FKD" --output-fkd "$EXP/fkd/$method" --teacher "$TEACHER" --ipc 3 --classes 100 \
  --dataset-name A_imsize224 --epochs 400 --batch-size 20 --workers 8 --seed 42 --fkd-seed 42 \
  --mean .4865 .5177 .5425 --std .2124 .2051 .2375 > "$EXP/logs/relabel_$method.log" 2>&1
 python "$ROOT/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$EXP/fkd/$method" --images 300 --classes 100 \
  --batch-size 20 --epochs 400 --output "$EXP/audits/${method}_fkd.json" >> "$EXP/logs/relabel_$method.log" 2>&1
 python "$ROOT/CV-DD/fine_grained/audit_fkd_metadata_alignment.py" --reference "$SOURCE_FKD" --candidate "$EXP/fkd/$method" \
  --output "$EXP/audits/${method}_metadata.json" >> "$EXP/logs/relabel_$method.log" 2>&1
}
relabel random_regions 0 & p0=$!
relabel fg_regions 1 & p1=$!
failed=0; wait "$p0" || failed=1; wait "$p1" || failed=1; ((failed==0))
date --iso-8601=seconds > "$EXP/status/relabel.complete"

student(){
 local method=$1 seed=$2 gpu=$3 image_root
 image_root="$SOURCE_EXP/construction/selected/$method/ipc3"
 mkdir -p "$EXP/results/$method" "$EXP/post_eval/$method/sseed$seed"
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 \
  --exp-name "deco_rrc_${method}_s${seed}" --original-data-path "$image_root" --fkd-path "$EXP/fkd/$method" \
  --output-dir "$EXP/post_eval/$method/sseed$seed" --batch-size 20 --epochs 400 --dataset-name A_imsize224 \
  --gradient-accumulation-steps 2 --mix-type cutmix --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$seed" \
  --temperature 20 --student-initialization imagenet-v1 --student-protocol-name "standard_v2_deco_rrc_$method" \
  --adamw-weight-decay 1e-5 --adamw-beta1 .9 --adamw-beta2 .999 --adamw-eps 1e-8 \
  --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 \
  --val-dir "$TEST" --disable-wandb --per-class-output "$EXP/results/$method/ipc3_sseed$seed.json" > "$EXP/logs/eval_${method}_s${seed}.log" 2>&1
}
pids=(); index=0
for method in random_regions fg_regions; do for seed in 42 43 44; do
 student "$method" "$seed" $((index%2)) & pids+=("$!"); index=$((index+1))
 if ((${#pids[@]}==4)); then for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; pids=(); fi
done; done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; ((failed==0))
python "$ROOT/CV-DD/fine_grained/summarize_deco_style_aircraft.py" --root "$EXP" --r0-results "$R0_RESULTS" \
 --construction-manifest "$CONSTRUCTION" --protocol controlled_deco_style_regions_rrc_cutmix_v1 > "$EXP/logs/summary.log" 2>&1
python - "$EXP/experiment_definition.json" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]);x=json.loads(p.read_text());x['status']='complete';t=p.with_suffix('.json.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
PY
rm -f "$EXP/status/running"; date --iso-8601=seconds > "$EXP/status/complete"
