#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOFT_FULL=/linxi/dataset/FGDD_DeCO_style/aircraft_ipc3_seed0_v1
SOFT_RRC=/linxi/dataset/FGDD_DeCO_style/aircraft_ipc3_seed0_rrc_v1
EXP="${EXP_ROOT:-/linxi/dataset/FGDD_DeCO_style/aircraft_ipc3_seed0_hard_factorial_v1}"
R0_FULL=/linxi/dataset/FG_HardAugFactorial_v2/aircraft_ipc3_v1/results/fullframe_cutmix/random_real/source_seed0
R0_RRC=/linxi/dataset/FG_HardReplay_standard/v2/aircraft_v1/results/random_real/source_seed0
TEST=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/test
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$EXP"/{logs,status,locks,results,post_eval,summary,audits}
exec 9>"$EXP/locks/launcher.lock"; flock -n 9 || exit 75
trap 's=$?; if ((s)); then rm -f "$EXP/status/running"; echo "$(date --iso-8601=seconds) exit=$s" > "$EXP/status/failed"; fi' EXIT
rm -f "$EXP/status/failed" "$EXP/status/complete"; date --iso-8601=seconds > "$EXP/status/running"

python - "$SOFT_FULL" "$SOFT_RRC" "$EXP/experiment_definition.json" "$(git -C "$ROOT" rev-parse HEAD)" <<'PY'
import json,os,sys
from pathlib import Path
full,rrc,out=map(Path,sys.argv[1:4]); revision=sys.argv[4]
for root in (full,rrc):
 for method in ('random_regions','fg_regions'):
  a=json.loads((root/f'audits/{method}_metadata.json').read_text()); assert a['status']=='complete' and a['total_mismatches']==0
x={'status':'running','protocol':'controlled_deco_style_regions_hard_rrc_factorial_v1','git_revision':revision,
   'views':{'fullframe_cutmix':str(full),'rrc_cutmix':str(rrc)},'methods':['random_regions','fg_regions'],
   'student_seeds':[42,43,44],'new_fkd_sets':0,'new_student_runs':12,
   'target':'ground-truth class T1 CE mixed by actual CutMix bbox area through --fkd-hard-label',
   'invariants':'same image trees and exact cached sampler/RRC/flip/CutMix metadata as Soft; only target changes'}
t=out.with_suffix('.json.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,out)
PY

images(){ echo "$SOFT_FULL/construction/selected/$2/ipc3"; }
fkd(){ if [[ $1 == fullframe_cutmix ]]; then echo "$SOFT_FULL/fkd/$2"; else echo "$SOFT_RRC/fkd/$2"; fi; }
student(){
 local view=$1 method=$2 seed=$3 gpu=$4 result
 result="$EXP/results/$view/$method/ipc3_sseed$seed.json"
 mkdir -p "$(dirname "$result")" "$EXP/post_eval/$view/$method/sseed$seed"
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 \
  --exp-name "deco_hard_${view}_${method}_s${seed}" --original-data-path "$(images "$view" "$method")" --fkd-path "$(fkd "$view" "$method")" \
  --output-dir "$EXP/post_eval/$view/$method/sseed$seed" --batch-size 20 --epochs 400 --dataset-name A_imsize224 \
  --gradient-accumulation-steps 2 --mix-type cutmix --fkd-hard-label --workers 8 --persistent-workers \
  --fkd_seed 42 --train-seed "$seed" --temperature 20 --student-initialization imagenet-v1 \
  --student-protocol-name "standard_v2_deco_hard_${view}_${method}" --adamw-weight-decay 1e-5 \
  --adamw-beta1 .9 --adamw-beta2 .999 --adamw-eps 1e-8 --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 \
  --cosine-t-max 400 --cosine-eta-min 0 --val-dir "$TEST" --disable-wandb --per-class-output "$result" \
  > "$EXP/logs/eval_${view}_${method}_s${seed}.log" 2>&1
}
failed=0; pids=(); index=0
for view in fullframe_cutmix rrc_cutmix; do for method in random_regions fg_regions; do for seed in 42 43 44; do
 student "$view" "$method" "$seed" $((index%2)) & pids+=("$!"); index=$((index+1))
 if ((${#pids[@]}==4)); then for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; pids=(); fi
done; done; done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; ((failed==0))
python "$ROOT/CV-DD/fine_grained/summarize_deco_style_aircraft_hard_factorial.py" --root "$EXP" \
 --r0-full-results "$R0_FULL" --r0-rrc-results "$R0_RRC" > "$EXP/logs/summary.log" 2>&1
python - "$EXP/experiment_definition.json" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]);x=json.loads(p.read_text());x['status']='complete';t=p.with_suffix('.json.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
PY
rm -f "$EXP/status/running"; date --iso-8601=seconds > "$EXP/status/complete"
