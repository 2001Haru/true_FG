#!/usr/bin/env bash
set -euo pipefail
ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EXP="${EXP_ROOT:-/linxi/dataset/FGDD_six_source/aircraft_ipc3_bbox_pack_v1}"
STAGE="${STAGE_ROOT:-/tmp/fgdd_aircraft_six_source_v1}"
SELECTION=/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard/manifests/A_imsize224/rseed0/ipc10.json
RAW=/linxi/dataset/FD2/raw/fgvc-aircraft-2013b/data
TEACHER=/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42/ResNet18.pth
TEST=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/test
WEIGHTS=/linxi/models/torchvision/resnet18-f37072fd.pth
MANIFEST="$EXP/construction/six_source_manifest.json"; PACKED="$EXP/construction/packed"
REF_FKD="$STAGE/fkd/reference"; COMP_FKD="$STAGE/fkd/compressed"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$EXP"/{construction,logs,status,locks,fkd_archives,results,checkpoints,post_eval,audits,summary} "$STAGE/fkd"
exec 9>"$EXP/locks/launcher.lock"; flock -n 9 || exit 75
trap 's=$?; if ((s)); then rm -f "$EXP/status/running"; echo "$(date --iso-8601=seconds) exit=$s" > "$EXP/status/failed"; fi' EXIT
rm -f "$EXP/status/failed" "$EXP/status/complete"; date --iso-8601=seconds > "$EXP/status/running"

if [[ ! -f "$MANIFEST" ]]; then
 python -u "$ROOT/CV-DD/fine_grained/prepare_aircraft_six_source_pack.py" --selection-manifest "$SELECTION" \
  --raw-images "$RAW/images" --boxes "$RAW/images_box.txt" --output-root "$EXP/construction" \
  > "$EXP/logs/construction.log" 2>&1
fi
python - "$MANIFEST" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));assert x['status']=='complete' and x['parents']==300 and x['unique_sources']==600
assert x['sources_per_class']==6 and all(len(r['sources'])==2 for r in x['records'])
for c in range(100):
 rows=[r for r in x['records'] if r['class_id']==c];assert len(rows)==3
 assert len({s['identity'] for r in rows for s in r['sources']})==6
print('construction gate passed')
PY
date --iso-8601=seconds > "$EXP/status/construction.complete"

if [[ ! -f "$REF_FKD/relabel_manifest.json" && -f "$EXP/fkd_archives/reference.tar" ]]; then mkdir -p "$REF_FKD" "$COMP_FKD"; tar -C "$REF_FKD" -xf "$EXP/fkd_archives/reference.tar"; tar -C "$COMP_FKD" -xf "$EXP/fkd_archives/compressed.tar"; fi
if [[ ! -f "$REF_FKD/relabel_manifest.json" ]]; then
 CUDA_VISIBLE_DEVICES=0 python -u "$ROOT/CV-DD/fine_grained/relabel_aircraft_six_source.py" --manifest "$MANIFEST" \
  --reference-fkd "$REF_FKD" --compressed-fkd "$COMP_FKD" --teacher "$TEACHER" --epochs 400 --batch-size 20 \
  --workers 8 --seed 42 > "$EXP/logs/relabel.log" 2>&1
fi
python "$ROOT/CV-DD/fine_grained/audit_aircraft_six_source_fkd.py" --manifest "$MANIFEST" --reference "$REF_FKD" \
 --compressed "$COMP_FKD" --output "$EXP/audits/soft_fkd.json" > "$EXP/logs/fkd_audit.log" 2>&1
if [[ ! -f "$EXP/fkd_archives/reference.tar" ]]; then
 tar -C "$REF_FKD" -cf "$EXP/fkd_archives/reference.tar.tmp" .; mv "$EXP/fkd_archives/reference.tar.tmp" "$EXP/fkd_archives/reference.tar"
 tar -C "$COMP_FKD" -cf "$EXP/fkd_archives/compressed.tar.tmp" .; mv "$EXP/fkd_archives/compressed.tar.tmp" "$EXP/fkd_archives/compressed.tar"
fi
date --iso-8601=seconds > "$EXP/status/relabel.complete"

soft(){
 local mode=$1 seed=$2 gpu=$3 fkd result="$EXP/results/soft/$mode/sseed${seed}.json"
 [[ "$mode" == reference ]] && fkd="$REF_FKD" || fkd="$COMP_FKD"
 mkdir -p "$(dirname "$result")" "$EXP/post_eval/soft/$mode/sseed$seed"
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 \
  --exp-name "six_source_soft_${mode}_s${seed}" --original-data-path "$PACKED" --fkd-path "$fkd" \
  --paired-source-manifest "$MANIFEST" --paired-source-mode "$mode" --output-dir "$EXP/post_eval/soft/$mode/sseed$seed" \
  --batch-size 20 --epochs 400 --dataset-name A_imsize224 --gradient-accumulation-steps 2 --mix-type cutmix \
  --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$seed" --temperature 20 --student-initialization imagenet-v1 \
  --student-protocol-name "standard_v2_six_source_fullframe_$mode" --adamw-weight-decay 1e-5 --adamw-beta1 .9 \
  --adamw-beta2 .999 --adamw-eps 1e-8 --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 \
  --cosine-t-max 400 --cosine-eta-min 0 --val-dir "$TEST" --disable-wandb --per-class-output "$result" \
  > "$EXP/logs/soft_${mode}_s${seed}.log" 2>&1
}

hard(){
 local mode=$1 seed=$2 gpu=$3 result="$EXP/results/hard/$mode/sseed${seed}.json"
 mkdir -p "$(dirname "$result")" "$EXP/checkpoints/hard/$mode/sseed$seed"
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_hard_label_v1.py" --paired-source-manifest "$MANIFEST" \
  --paired-source-mode "$mode" --val-dir "$TEST" --dataset-name A_imsize224 --num-classes 100 --ipc 3 \
  --student-seed "$seed" --result "$result" --checkpoint-dir "$EXP/checkpoints/hard/$mode/sseed$seed" \
  --imagenet-weights-path "$WEIGHTS" --total-updates 3000 --batch-size 64 --backbone-lr 3e-4 --head-lr 3e-3 \
  --backbone-min-lr 0 --head-min-lr 0 --momentum .9 --weight-decay 5e-4 --eval-every-updates 300 \
  --workers 8 --persistent-workers --val-batch-size 256 --protocol-name "hard_label_v1_six_source_mild_rc_$mode" \
  --train-crop-mode mild_rc > "$EXP/logs/hard_${mode}_s${seed}.log" 2>&1
}

failed=0;pids=();index=0
for protocol in soft hard; do for mode in reference compressed; do for seed in 42 43 44; do
 "$protocol" "$mode" "$seed" $((index%2)) & pids+=("$!");index=$((index+1))
 if ((${#pids[@]}==4)); then for pid in "${pids[@]}"; do wait "$pid"||failed=1;done;pids=();fi
done;done;done
for pid in "${pids[@]}";do wait "$pid"||failed=1;done;((failed==0))
python "$ROOT/CV-DD/fine_grained/summarize_aircraft_six_source.py" --root "$EXP" > "$EXP/logs/summary.log" 2>&1
rm -f "$EXP/status/running";date --iso-8601=seconds > "$EXP/status/complete"
