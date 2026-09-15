#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SELECTION_ROOT=/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard
DATA=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224
TEACHER_DIR=/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42
EXP="${EXP_ROOT:-/linxi/dataset/FG_RandomReal_soft_v2/aircraft_fullframe_ipc5_10_20_v1}"
IPCS=(5 10 20); RSEEDS=(0 1 2); SSEEDS=(42 43 44)
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_HOME=/linxi/models/torchvision_cache
mkdir -p "$EXP"/{logs,status,locks,fkd,results,post_eval,audits,summary}
exec 9>"$EXP/locks/launcher.lock"; flock -n 9 || exit 75
trap 's=$?; if ((s)); then rm -f "$EXP/status/running"; echo "$(date --iso-8601=seconds) exit=$s" > "$EXP/status/failed"; fi' EXIT
rm -f "$EXP/status/failed" "$EXP/status/complete"; date --iso-8601=seconds > "$EXP/status/running"

for rseed in "${RSEEDS[@]}"; do for ipc in "${IPCS[@]}"; do
 selected="$SELECTION_ROOT/selected/A_imsize224/rseed$rseed/ipc$ipc"
 manifest="$SELECTION_ROOT/manifests/A_imsize224/rseed$rseed/ipc$ipc.json"
 if [[ ! -f "$manifest" ]]; then
  python "$ROOT/CV-DD/fine_grained/prepare_random_real_fg.py" --data-dir "$DATA" \
   --output-dir "$selected" --manifest "$manifest" --dataset-name A_imsize224 --classes 100 --ipc "$ipc" \
   --selection-seed "$rseed" --link-mode symlink > "$EXP/logs/selection_ipc${ipc}_rseed${rseed}.log" 2>&1
 fi
done; done
python "$ROOT/CV-DD/fine_grained/audit_random_real_nested_scaling.py" --selection-root "$SELECTION_ROOT" \
 --output "$EXP/audits/nested_selection.json" > "$EXP/logs/nested_selection.log" 2>&1
date --iso-8601=seconds > "$EXP/status/selection.complete"

python - "$EXP/matrix_definition.json" "$SELECTION_ROOT" <<'PY'
import json,os,sys
from pathlib import Path
out,selection=map(Path,sys.argv[1:])
x={'status':'running','protocol':'random_real_soft_v2_fullframe_cutmix_scaling_v1','dataset':'A_imsize224',
   'ipcs':[5,10,20],'selection_seeds':[0,1,2],'student_seeds':[42,43,44],
   'nesting':'IPC3 strict prefix IPC5 strict prefix IPC10 strict prefix IPC20 within each selection seed',
   'selection_root':str(selection.resolve()),'teacher_seed':42,'new_fkd_sets':9,
   'new_soft_student_runs':27,'new_hard_v1_student_runs':27,'total_new_student_runs':54,
   'view':'full 224 image + horizontal flip + CutMix; RRC disabled',
   'student_protocol':'standard v2 ImageNet-V1 ResNet18; T20 KL no T^2; batch20 accumulation2; 400 epochs'}
tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');os.replace(tmp,out)
PY

selected(){ echo "$SELECTION_ROOT/selected/A_imsize224/rseed$2/ipc$1"; }
manifest(){ echo "$SELECTION_ROOT/manifests/A_imsize224/rseed$2/ipc$1.json"; }
fkd(){ echo "$EXP/fkd/rseed$2/ipc${1}_bs20_ipc$1"; }
relabel(){
 local ipc=$1 rseed=$2 gpu=$3 base="$EXP/fkd/rseed$rseed/ipc$ipc" actual expected count=0
 actual="${base}_bs20_ipc${ipc}"; expected=$((400*100*ipc/20))
 [[ -d "$actual" ]] && count=$(find "$actual" -name 'batch_*.tar' -type f | wc -l)
 if ((count!=expected)); then
  CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/relabel/relabel.py" --syn-data-path "$(selected "$ipc" "$rseed")" \
   --fkd-path "$base" --model-pool-dir "$TEACHER_DIR" --teacher-model-name ResNet18 --gpu 0 --batch-size 20 \
   --workers 8 --dataset-name A_imsize224 --epochs 400 --seed 42 --fkd-seed 42 --full-image-resize \
   --min-scale-crops 1 --max-scale-crops 1 --mix-type cutmix --use-fp16 --mode fkd_save \
   > "$EXP/logs/relabel_ipc${ipc}_rseed${rseed}.log" 2>&1
 fi
 audit="$actual/fkd_audit.json"
 central_audit="$EXP/audits/fkd_ipc${ipc}_rseed${rseed}.json"
 if [[ ! -f "$audit" && -f "$central_audit" ]] && python -c "import json; assert json.load(open('$central_audit'))['status']=='complete'"; then
  cp "$central_audit" "$audit"
 fi
 if [[ ! -f "$audit" ]] || ! python -c "import json; assert json.load(open('$audit'))['status']=='complete'"; then
  python "$ROOT/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$actual" --images $((100*ipc)) --classes 100 \
   --batch-size 20 --epochs 400 --output "$audit" >> "$EXP/logs/relabel_ipc${ipc}_rseed${rseed}.log" 2>&1
 fi
}
failed=0; pids=(); index=0
for ipc in "${IPCS[@]}"; do for rseed in "${RSEEDS[@]}"; do
 relabel "$ipc" "$rseed" $((index%2)) & pids+=("$!"); index=$((index+1))
 if ((${#pids[@]}==2)); then for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; pids=(); fi
done; done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; ((failed==0))
date --iso-8601=seconds > "$EXP/status/relabel.complete"

student(){
 local ipc=$1 rseed=$2 sseed=$3 gpu=$4 result="$EXP/results/rseed$rseed/ipc${ipc}_sseed${sseed}.json"
 mkdir -p "$(dirname "$result")" "$EXP/post_eval/rseed$rseed/ipc$ipc/sseed$sseed"
 if [[ ! -f "$result" ]]; then
  CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc "$ipc" \
   --exp-name "random_real_fullframe_ipc${ipc}_r${rseed}_s${sseed}" --original-data-path "$(selected "$ipc" "$rseed")" \
   --fkd-path "$(fkd "$ipc" "$rseed")" --output-dir "$EXP/post_eval/rseed$rseed/ipc$ipc/sseed$sseed" \
   --batch-size 20 --epochs 400 --dataset-name A_imsize224 --gradient-accumulation-steps 2 --mix-type cutmix \
   --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$sseed" --temperature 20 \
   --student-initialization imagenet-v1 --student-protocol-name standard_protocol_v2 --adamw-weight-decay 1e-5 \
   --adamw-beta1 .9 --adamw-beta2 .999 --adamw-eps 1e-8 --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 \
   --cosine-t-max 400 --cosine-eta-min 0 --val-dir "$DATA/test" --disable-wandb --per-class-output "$result" \
   > "$EXP/logs/eval_ipc${ipc}_rseed${rseed}_sseed${sseed}.log" 2>&1
 fi
 python "$ROOT/CV-DD/fine_grained/record_random_real_soft_v2_result.py" --result "$result" \
  --selection-manifest "$(manifest "$ipc" "$rseed")" --teacher-dir "$TEACHER_DIR" --fkd-dir "$(fkd "$ipc" "$rseed")" \
  --ipc "$ipc" --selection-seed "$rseed" --student-seed "$sseed" >> "$EXP/logs/eval_ipc${ipc}_rseed${rseed}_sseed${sseed}.log" 2>&1
}
pids=(); index=0
for ipc in "${IPCS[@]}"; do for rseed in "${RSEEDS[@]}"; do for sseed in "${SSEEDS[@]}"; do
 student "$ipc" "$rseed" "$sseed" $((index%2)) & pids+=("$!"); index=$((index+1))
 if ((${#pids[@]}==4)); then for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; pids=(); fi
done; done; done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; ((failed==0))
python "$ROOT/CV-DD/fine_grained/summarize_random_real_fullframe_scaling.py" --root "$EXP" > "$EXP/logs/summary.log" 2>&1
date --iso-8601=seconds > "$EXP/status/soft.complete"

hard_student(){
 local ipc=$1 rseed=$2 sseed=$3 gpu=$4 result="$EXP/hard_v1/results/rseed$rseed/ipc${ipc}_sseed${sseed}.json"
 mkdir -p "$(dirname "$result")" "$EXP/hard_v1/checkpoints/rseed$rseed/ipc$ipc/sseed$sseed"
 if [[ ! -f "$result" ]]; then
  CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_hard_label_v1.py" --train-dir "$(selected "$ipc" "$rseed")" \
   --val-dir "$DATA/test" --dataset-name A_imsize224 --num-classes 100 --ipc "$ipc" --student-seed "$sseed" \
   --result "$result" --checkpoint-dir "$EXP/hard_v1/checkpoints/rseed$rseed/ipc$ipc/sseed$sseed" \
   --imagenet-weights-path /linxi/models/torchvision/resnet18-f37072fd.pth --total-updates 3000 --batch-size 64 \
   --backbone-lr 3e-4 --head-lr 3e-3 --backbone-min-lr 0 --head-min-lr 0 --momentum .9 --weight-decay 5e-4 \
   --eval-every-updates 300 --workers 8 --persistent-workers --val-batch-size 256 --protocol-name hard_label_v1_mild_rc \
   --train-crop-mode mild_rc > "$EXP/logs/hard_ipc${ipc}_rseed${rseed}_sseed${sseed}.log" 2>&1
 fi
 python "$ROOT/CV-DD/fine_grained/audit_hard_label_v1_result.py" --result "$result" --dataset A_imsize224 --classes 100 \
  --ipc "$ipc" --student-seed "$sseed" --validation-images 3333 --protocol-name hard_label_v1_mild_rc \
  --train-crop-mode mild_rc --rrc-min-scale .08 --rrc-max-scale 1 >> "$EXP/logs/hard_ipc${ipc}_rseed${rseed}_sseed${sseed}.log" 2>&1
}
pids=(); index=0
for ipc in "${IPCS[@]}"; do for rseed in "${RSEEDS[@]}"; do for sseed in "${SSEEDS[@]}"; do
 hard_student "$ipc" "$rseed" "$sseed" $((index%2)) & pids+=("$!"); index=$((index+1))
 if ((${#pids[@]}==4)); then for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; pids=(); fi
done; done; done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; ((failed==0))
python "$ROOT/CV-DD/fine_grained/summarize_random_real_hard_v1_scaling.py" --root "$EXP" > "$EXP/logs/hard_summary.log" 2>&1
date --iso-8601=seconds > "$EXP/status/hard_v1.complete"
python - "$EXP/matrix_definition.json" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]);x=json.loads(p.read_text());x['status']='complete';t=p.with_suffix('.json.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
PY
rm -f "$EXP/status/running"; date --iso-8601=seconds > "$EXP/status/complete"
