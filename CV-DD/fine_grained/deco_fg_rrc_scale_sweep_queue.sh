#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE=/linxi/dataset/FGDD_DeCO_style/aircraft_ipc3_seed0_v1
BASELINE=/linxi/dataset/FGDD_DeCO_style/aircraft_ipc3_seed0_rrc_v1
EXP="${EXP_ROOT:-/linxi/dataset/FGDD_DeCO_style/aircraft_ipc3_seed0_fg_rrc_scale_sweep_v1}"
IMAGES="$SOURCE/construction/selected/fg_regions/ipc3"
TEACHER_DIR=/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42
TEST=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/test
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$EXP"/{logs,status,locks,fkd,results,post_eval,audits,summary}
exec 9>"$EXP/locks/launcher.lock"; flock -n 9 || exit 75
trap 's=$?; if ((s)); then rm -f "$EXP/status/running"; echo "$(date --iso-8601=seconds) exit=$s" > "$EXP/status/failed"; fi' EXIT
rm -f "$EXP/status/failed" "$EXP/status/complete"; date --iso-8601=seconds > "$EXP/status/running"

python - "$SOURCE/construction/construction_manifest.json" "$BASELINE/summary/summary.json" "$EXP/experiment_definition.json" "$(git -C "$ROOT" rev-parse HEAD)" <<'PY'
import json,os,sys
from pathlib import Path
construction,baseline,out=map(Path,sys.argv[1:4]); revision=sys.argv[4]
c=json.loads(construction.read_text()); b=json.loads(baseline.read_text())
assert c['status']=='complete' and c['stored_images']==300 and c['regions']==1200
assert b['status']=='complete' and b['protocol']=='controlled_deco_style_regions_rrc_cutmix_v1'
x={'status':'running','protocol':'deco_fg_regions_soft_v2_rrc_scale_sweep_v1','git_revision':revision,
   'image_root':str(Path('/linxi/dataset/FGDD_DeCO_style/aircraft_ipc3_seed0_v1/construction/selected/fg_regions/ipc3')),
   'baseline_rrc_range':[0.08,1.0],'new_rrc_ranges':[[0.1,0.7],[0.3,1.0],[0.5,1.0]],
   'student_seeds':[42,43,44],'new_fkd_sets':3,'new_student_runs':9,
   'invariants':'same FG-region images, Teacher42 train-mode BSSL, flip, CutMix alpha1, FKD seed42, Soft standard v2 Student; only RRC scale range changes'}
t=out.with_suffix('.json.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,out)
PY

scale(){ case "$1" in rrc_010_070) echo '0.1 0.7';; rrc_030_100) echo '0.3 1.0';; rrc_050_100) echo '0.5 1.0';; *) return 2;; esac; }
fkd(){ echo "$EXP/fkd/$1/ipc3_bs20_ipc3"; }
relabel(){
 local arm=$1 gpu=$2 lo hi base actual
 read -r lo hi < <(scale "$arm"); base="$EXP/fkd/$arm/ipc3"; actual="$(fkd "$arm")"
 local count=0 status=''
 [[ -d "$actual" ]] && count=$(find "$actual" -type f -name 'batch_*.tar' | wc -l)
 [[ -f "$actual/relabel_manifest.json" ]] && status=$(python -c "import json;print(json.load(open('$actual/relabel_manifest.json')).get('status',''))")
 if [[ "$count" != 6000 || "$status" != complete ]]; then
  CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/relabel/relabel.py" --syn-data-path "$IMAGES" --fkd-path "$base" \
   --model-pool-dir "$TEACHER_DIR" --teacher-model-name ResNet18 --gpu 0 --batch-size 20 --workers 8 \
   --dataset-name A_imsize224 --epochs 400 --seed 42 --fkd-seed 42 --use-fp16 --mode fkd_save \
   --min-scale-crops "$lo" --max-scale-crops "$hi" --mix-type cutmix > "$EXP/logs/relabel_$arm.log" 2>&1
 fi
 python "$ROOT/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$actual" --images 300 --classes 100 --batch-size 20 \
  --epochs 400 --output "$EXP/audits/${arm}_fkd.json" >> "$EXP/logs/relabel_$arm.log" 2>&1
 python - "$actual/relabel_manifest.json" "$lo" "$hi" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['status']=='complete' and x['mix_type']=='cutmix'
assert abs(x['min_scale_crops']-float(sys.argv[2]))<1e-12 and abs(x['max_scale_crops']-float(sys.argv[3]))<1e-12
PY
}
failed=0
relabel rrc_010_070 0 & p0=$!; relabel rrc_030_100 1 & p1=$!
wait "$p0" || failed=1; wait "$p1" || failed=1; ((failed==0))
relabel rrc_050_100 0; date --iso-8601=seconds > "$EXP/status/relabel.complete"

student(){
 local arm=$1 seed=$2 gpu=$3 result
 result="$EXP/results/$arm/ipc3_sseed$seed.json"; mkdir -p "$(dirname "$result")" "$EXP/post_eval/$arm/sseed$seed"
 if [[ ! -f "$result" ]]; then CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 \
  --exp-name "deco_fg_${arm}_s${seed}" --original-data-path "$IMAGES" --fkd-path "$(fkd "$arm")" \
  --output-dir "$EXP/post_eval/$arm/sseed$seed" --batch-size 20 --epochs 400 --dataset-name A_imsize224 \
  --gradient-accumulation-steps 2 --mix-type cutmix --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$seed" \
  --temperature 20 --student-initialization imagenet-v1 --student-protocol-name "standard_v2_deco_fg_${arm}" \
  --adamw-weight-decay 1e-5 --adamw-beta1 .9 --adamw-beta2 .999 --adamw-eps 1e-8 \
  --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 \
  --val-dir "$TEST" --disable-wandb --per-class-output "$result" > "$EXP/logs/eval_${arm}_s${seed}.log" 2>&1; fi
}
pids=(); index=0
for arm in rrc_010_070 rrc_030_100 rrc_050_100; do for seed in 42 43 44; do
 student "$arm" "$seed" $((index%2)) & pids+=("$!"); index=$((index+1))
 if ((${#pids[@]}==4)); then for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; pids=(); fi
done; done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; ((failed==0))
python "$ROOT/CV-DD/fine_grained/summarize_deco_fg_rrc_scale_sweep.py" --root "$EXP" \
 --baseline-results "$BASELINE/results/fg_regions" > "$EXP/logs/summary.log" 2>&1
python - "$EXP/experiment_definition.json" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]);x=json.loads(p.read_text());x['status']='complete';t=p.with_suffix('.json.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
PY
rm -f "$EXP/status/running"; date --iso-8601=seconds > "$EXP/status/complete"
