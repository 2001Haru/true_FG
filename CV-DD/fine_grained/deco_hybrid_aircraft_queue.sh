#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE=/linxi/dataset/FGDD_DeCO_style/aircraft_ipc3_seed0_v1
R0_ROOT=/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard
EXP="${EXP_ROOT:-/linxi/dataset/FGDD_hybrid/aircraft_ipc3_seed0_v1}"
TEACHER_DIR=/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42
TEACHER="$TEACHER_DIR/ResNet18.pth"
TEST=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/test
R0_RESULTS=/linxi/dataset/FG_SoftAugFactorial_v2/aircraft_ipc3_v1/results/fullframe_cutmix/random_real/source_seed0
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$EXP"/{construction,logs,status,locks,fkd,results,post_eval,audits,summary}
exec 9>"$EXP/locks/launcher.lock"; flock -n 9 || exit 75
trap 's=$?; if ((s)); then rm -f "$EXP/status/running"; echo "$(date --iso-8601=seconds) exit=$s" > "$EXP/status/failed"; fi' EXIT
rm -f "$EXP/status/failed" "$EXP/status/complete"; date --iso-8601=seconds > "$EXP/status/running"

python "$ROOT/CV-DD/fine_grained/prepare_deco_hybrid_aircraft.py" \
 --construction-manifest "$SOURCE/construction/construction_manifest.json" \
 --r0-selection-manifest "$R0_ROOT/manifests/A_imsize224/rseed0/ipc3.json" \
 --output-root "$EXP/construction" > "$EXP/logs/construction.log" 2>&1
python - "$EXP/construction/construction_manifest.json" "$EXP/experiment_definition.json" "$(git -C "$ROOT" rev-parse HEAD)" <<'PY'
import json,os,sys
from pathlib import Path
construction,out=map(Path,sys.argv[1:3]);revision=sys.argv[3];c=json.loads(construction.read_text())
assert c['status']=='complete' and c['stored_images']==300 and c['independent_sources_per_class']==9
for row in c['classes_detail']:
 assert len(row['independent_source_ids'])==len(set(row['independent_source_ids']))==9
x={'status':'running','protocol':'deco_hybrid_one_full_two_mosaics_soft_v2','git_revision':revision,
   'construction_manifest':str(construction),'groups':{
    'hybrid_fg_uniform_rrc':'one R0 full + two FG mosaics; all RRC',
    'hybrid_fg_typed':'same images; full slot full-frame, mosaics RRC',
    'hybrid_random_typed':'same source assignment; random-region mosaics; full slot full-frame, mosaics RRC'},
   'rrc_scale':[0.08,1.0],'student_seeds':[42,43,44],'new_fkd_sets':3,'new_student_runs':9,
   'storage':{'ipc':3,'full_slots':1,'mosaic_slots':2,'independent_sources_per_class':9}}
t=out.with_suffix('.json.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,out)
PY

FG_IMAGES="$EXP/construction/selected/hybrid_fg/ipc3"
RANDOM_IMAGES="$EXP/construction/selected/hybrid_random/ipc3"
BASE="$EXP/fkd/hybrid_fg_uniform_rrc/ipc3_bs20_ipc3"
FG_TYPED="$EXP/fkd/hybrid_fg_typed/ipc3_bs20_ipc3"
RANDOM_TYPED="$EXP/fkd/hybrid_random_typed/ipc3_bs20_ipc3"

fkd_complete(){
 local path=$1 count=0 status=''
 [[ -d "$path" ]] && count=$(find "$path" -type f -name 'batch_*.tar' | wc -l)
 [[ -f "$path/relabel_manifest.json" ]] && status=$(python -c "import json;print(json.load(open('$path/relabel_manifest.json')).get('status',''))")
 [[ "$count" == 6000 && "$status" == complete ]]
}
if ! fkd_complete "$BASE"; then
 CUDA_VISIBLE_DEVICES=0 python -u "$ROOT/CV-DD/relabel/relabel.py" --syn-data-path "$FG_IMAGES" \
  --fkd-path "$EXP/fkd/hybrid_fg_uniform_rrc/ipc3" --model-pool-dir "$TEACHER_DIR" --teacher-model-name ResNet18 \
  --gpu 0 --batch-size 20 --workers 8 --dataset-name A_imsize224 --epochs 400 --seed 42 --fkd-seed 42 \
  --use-fp16 --mode fkd_save --min-scale-crops .08 --max-scale-crops 1 --mix-type cutmix > "$EXP/logs/relabel_hybrid_fg_uniform_rrc.log" 2>&1
fi
python "$ROOT/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$BASE" --images 300 --classes 100 --batch-size 20 \
 --epochs 400 --output "$EXP/audits/hybrid_fg_uniform_rrc_fkd.json" >> "$EXP/logs/relabel_hybrid_fg_uniform_rrc.log" 2>&1

rescore_typed(){
 local method=$1 image_root=$2 output=$3 gpu=$4
 if ! fkd_complete "$output"; then
  CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/fine_grained/rescore_fkd_views.py" --image-root "$image_root" \
   --source-fkd "$BASE" --output-fkd "$output" --teacher "$TEACHER" --ipc 3 --classes 100 --dataset-name A_imsize224 \
   --epochs 400 --batch-size 20 --workers 8 --seed 42 --fkd-seed 42 --mean .4865 .5177 .5425 --std .2124 .2051 .2375 \
   --force-full-prefix slot0_full > "$EXP/logs/relabel_${method}.log" 2>&1
 fi
 python "$ROOT/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$output" --images 300 --classes 100 --batch-size 20 \
  --epochs 400 --output "$EXP/audits/${method}_fkd.json" >> "$EXP/logs/relabel_${method}.log" 2>&1
 python "$ROOT/CV-DD/fine_grained/audit_hybrid_typed_fkd.py" --source "$BASE" --candidate "$output" \
  --output "$EXP/audits/${method}_typed_geometry.json" >> "$EXP/logs/relabel_${method}.log" 2>&1
}
failed=0
rescore_typed hybrid_fg_typed "$FG_IMAGES" "$FG_TYPED" 0 & p0=$!
rescore_typed hybrid_random_typed "$RANDOM_IMAGES" "$RANDOM_TYPED" 1 & p1=$!
wait "$p0" || failed=1; wait "$p1" || failed=1; ((failed==0))
python "$ROOT/CV-DD/fine_grained/audit_fkd_metadata_alignment.py" --reference "$FG_TYPED" --candidate "$RANDOM_TYPED" \
 --output "$EXP/audits/typed_fg_random_metadata_alignment.json" > "$EXP/logs/typed_alignment.log" 2>&1
date --iso-8601=seconds > "$EXP/status/relabel.complete"

images(){ case "$1" in hybrid_random_typed) echo "$RANDOM_IMAGES";; *) echo "$FG_IMAGES";; esac; }
fkd(){ case "$1" in hybrid_fg_uniform_rrc) echo "$BASE";; hybrid_fg_typed) echo "$FG_TYPED";; hybrid_random_typed) echo "$RANDOM_TYPED";; *) return 2;; esac; }
student(){
 local method=$1 seed=$2 gpu=$3 result
 result="$EXP/results/$method/ipc3_sseed$seed.json"; mkdir -p "$(dirname "$result")" "$EXP/post_eval/$method/sseed$seed"
 if [[ ! -f "$result" ]]; then CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 \
  --exp-name "deco_hybrid_${method}_s${seed}" --original-data-path "$(images "$method")" --fkd-path "$(fkd "$method")" \
  --output-dir "$EXP/post_eval/$method/sseed$seed" --batch-size 20 --epochs 400 --dataset-name A_imsize224 \
  --gradient-accumulation-steps 2 --mix-type cutmix --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$seed" \
  --temperature 20 --student-initialization imagenet-v1 --student-protocol-name "standard_v2_deco_hybrid_$method" \
  --adamw-weight-decay 1e-5 --adamw-beta1 .9 --adamw-beta2 .999 --adamw-eps 1e-8 \
  --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 \
  --val-dir "$TEST" --disable-wandb --per-class-output "$result" > "$EXP/logs/eval_${method}_s${seed}.log" 2>&1; fi
}
pids=();index=0
for method in hybrid_fg_uniform_rrc hybrid_fg_typed hybrid_random_typed; do for seed in 42 43 44; do
 student "$method" "$seed" $((index%2)) & pids+=("$!");index=$((index+1))
 if ((${#pids[@]}==4)); then for pid in "${pids[@]}"; do wait "$pid" || failed=1; done;pids=();fi
done;done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done;((failed==0))
python "$ROOT/CV-DD/fine_grained/summarize_deco_hybrid_aircraft.py" --root "$EXP" --r0-results "$R0_RESULTS" > "$EXP/logs/summary.log" 2>&1
python - "$EXP/experiment_definition.json" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]);x=json.loads(p.read_text());x['status']='complete';t=p.with_suffix('.json.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
PY
rm -f "$EXP/status/running";date --iso-8601=seconds > "$EXP/status/complete"
