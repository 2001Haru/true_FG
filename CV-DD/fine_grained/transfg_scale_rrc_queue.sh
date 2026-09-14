#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE=/linxi/dataset/FG_ViT_Teachers/aircraft_seed42_224_v1
EXP="$BASE/scale_rrc_v1"
IMAGES=/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard/selected/A_imsize224/rseed0/ipc3
REF=/linxi/dataset/FG_HardAugFactorial_v2/aircraft_ipc3_v1/fkd
CM="$BASE/fkd/transfg/ipc3_bs20_ipc3_fp32"
RRC="$REF/rrc_no_cutmix/random_real/source_seed0/ipc3_bs20_ipc3"
NEW="$EXP/fkd/rrc_only"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$EXP"/{logs,results,post_eval,audits,status}
exec 9>"$EXP/launcher.lock"
flock -n 9 || exit 75
trap 's=$?; if ((s)); then echo "$s" > "$EXP/status/failed"; fi' EXIT
python "$ROOT/CV-DD/fine_grained/match_fkd_entropy_temperature.py" --reference "$REF/fullframe_cutmix/random_real/source_seed0/ipc3_bs20_ipc3" --candidate "$CM" --output "$EXP/audits/temperature.json" > "$EXP/logs/temperature.log" 2>&1
TT=$(python -c "import json;print(json.load(open('$EXP/audits/temperature.json'))['teacher_temperature'])")
student(){
 local arm=$1 seed=$2 gpu=$3 fkd=$4 temp=$5 mix=$6
 local opts=()
 if [[ $mix == cutmix ]]; then opts=(--mix-type cutmix); fi
 mkdir -p "$EXP/results/$arm"
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_fkd.py" \
 --model ResNet18 --ipc 3 --exp-name "transfg_${arm}_s${seed}" --original-data-path "$IMAGES" --fkd-path "$fkd" \
 --output-dir "$EXP/post_eval/$arm/sseed$seed" --batch-size 20 --epochs 400 --dataset-name A_imsize224 \
 --gradient-accumulation-steps 2 "${opts[@]}" --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$seed" \
 --temperature 20 --teacher-temperature "$temp" --student-initialization imagenet-v1 --student-protocol-name "standard_v2_transfg_$arm" \
 --adamw-weight-decay 1e-5 --adamw-beta1 .9 --adamw-beta2 .999 --adamw-eps 1e-8 --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 \
 --cosine-t-max 400 --cosine-eta-min 0 --val-dir /linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/test \
 --disable-wandb --per-class-output "$EXP/results/$arm/ipc3_sseed$seed.json" > "$EXP/logs/${arm}_s$seed.log" 2>&1
}
scale_branch(){
 for s in 42 43 44; do student entropy_match "$s" 0 "$CM" "$TT" cutmix; done
}
rrc_branch(){
 CUDA_VISIBLE_DEVICES=1 python -u "$ROOT/CV-DD/fine_grained/rescore_fkd_vit_teacher.py" --kind transfg \
 --source-root "$ROOT/third_party/teacher_backbones/TransFG" --checkpoint "$BASE/teachers/transfg/final_step10000.pth" \
 --image-root "$IMAGES" --source-fkd "$RRC" --output-fkd "$NEW" --mix-type none --workers 8 > "$EXP/logs/relabel.log" 2>&1
 python "$ROOT/CV-DD/fine_grained/audit_fkd_metadata_alignment.py" --reference "$RRC" --candidate "$NEW" --output "$EXP/audits/rrc_metadata.json" > "$EXP/logs/metadata.log" 2>&1
 for s in 42 43 44; do student rrc_only "$s" 1 "$NEW" 20 none; done
}
scale_branch & p0=$!
rrc_branch & p1=$!
failed=0
wait "$p0" || failed=1
wait "$p1" || failed=1
((failed==0))
python - "$EXP" <<'PY'
import json,sys,statistics
from pathlib import Path
r=Path(sys.argv[1]); out={}
for arm in ('entropy_match','rrc_only'):
 rows=[json.load(open(r/'results'/arm/f'ipc3_sseed{s}.json')) for s in (42,43,44)]
 assert all(x['temperature']==20 and x['epochs']==400 for x in rows)
 out[arm]={k:{'mean':statistics.mean(x[k] for x in rows),'sd':statistics.stdev(x[k] for x in rows)} for k in ('best_top1','final_epoch_top1')}
 out[arm]['rows']=[{k:x[k] for k in ('student_seed','best_top1','final_epoch_top1','teacher_temperature')} for x in rows]
(r/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
PY
date --iso-8601=seconds > "$EXP/status/complete"
