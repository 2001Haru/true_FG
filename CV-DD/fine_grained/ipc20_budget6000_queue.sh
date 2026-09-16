#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCALING=/linxi/dataset/FG_RandomReal_soft_v2/aircraft_fullframe_ipc5_10_20_v1
SELECTION=/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard
IPC3=/linxi/dataset/FG_SoftAugFactorial_v2/aircraft_ipc3_v1/results/fullframe_cutmix/random_real/source_seed0
TEST=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/test
EXP="${EXP_ROOT:-/linxi/dataset/FG_RandomReal_soft_v2/aircraft_ipc20_budget6000_v1}"
RSEEDS=(0 1 2); SSEEDS=(42 43 44)
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_HOME=/linxi/models/torchvision_cache
mkdir -p "$EXP"/{logs,status,locks,results,post_eval,summary,audits}
exec 9>"$EXP/locks/launcher.lock"; flock -n 9 || exit 75
trap 's=$?; if ((s)); then rm -f "$EXP/status/running"; echo "$(date --iso-8601=seconds) exit=$s" > "$EXP/status/failed"; fi' EXIT
rm -f "$EXP/status/failed" "$EXP/status/complete"; date --iso-8601=seconds > "$EXP/status/running"

python - "$SCALING" "$SELECTION" "$EXP/audits/input.json" <<'PY'
import json,os,sys
from pathlib import Path
scaling,selection,out=map(Path,sys.argv[1:])
nest=json.loads((scaling/'audits/nested_selection.json').read_text()); assert nest['status']=='complete'
for rseed in (0,1,2):
 manifest=json.loads((selection/f'manifests/A_imsize224/rseed{rseed}/ipc20.json').read_text())
 fkd=scaling/f'fkd/rseed{rseed}/ipc20_bs20_ipc20'
 audit=json.loads((fkd/'fkd_audit.json').read_text()); relabel=json.loads((fkd/'relabel_manifest.json').read_text())
 assert manifest['status']=='complete' and manifest['ipc']==20 and manifest['selection_seed']==rseed
 assert audit['status']=='complete' and audit['batch_files']==40000 and audit['images']==2000
 assert relabel['status']=='complete' and relabel['epochs']==400 and relabel['batch_size']==20
x={'status':'complete','ipc':20,'selection_seeds':[0,1,2],'student_seeds':[42,43,44],
   'consumed_fkd_epochs':[0,59],'physical_batches':6000,'views':120000,'epochs':60,
   'gradient_accumulation_steps':2,'optimizer_steps':3000,'scheduler_t_max':60,
   'new_fkd_sets':0,'teacher_queries':0,'new_student_runs':9}
out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');os.replace(tmp,out)
print(json.dumps(x,indent=2))
PY

student(){
 local rseed=$1 sseed=$2 gpu=$3 result="$EXP/results/rseed$rseed/ipc20_sseed$sseed.json"
 images="$SELECTION/selected/A_imsize224/rseed$rseed/ipc20"
 fkd="$SCALING/fkd/rseed$rseed/ipc20_bs20_ipc20"
 mkdir -p "$(dirname "$result")" "$EXP/post_eval/rseed$rseed/sseed$sseed"
 if [[ ! -f "$result" ]]; then
  CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 20 \
   --exp-name "random_real_ipc20_budget6000_r${rseed}_s${sseed}" --original-data-path "$images" --fkd-path "$fkd" \
   --output-dir "$EXP/post_eval/rseed$rseed/sseed$sseed" --batch-size 20 --epochs 60 --dataset-name A_imsize224 \
   --gradient-accumulation-steps 2 --mix-type cutmix --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$sseed" \
   --temperature 20 --student-initialization imagenet-v1 --student-protocol-name standard_protocol_v2_budget6000 \
   --adamw-weight-decay 1e-5 --adamw-beta1 .9 --adamw-beta2 .999 --adamw-eps 1e-8 --adamw-backbone-lr 1e-4 \
   --adamw-head-lr 1e-3 --cosine-t-max 60 --cosine-eta-min 0 --val-dir "$TEST" --disable-wandb \
   --per-class-output "$result" > "$EXP/logs/eval_rseed${rseed}_sseed${sseed}.log" 2>&1
 fi
}
failed=0;pids=();index=0
for rseed in "${RSEEDS[@]}"; do for sseed in "${SSEEDS[@]}"; do
 student "$rseed" "$sseed" $((index%2)) & pids+=("$!");index=$((index+1))
 if ((${#pids[@]}==4)); then for pid in "${pids[@]}"; do wait "$pid" || failed=1; done;pids=();fi
done;done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done;((failed==0))
python "$ROOT/CV-DD/fine_grained/summarize_ipc20_budget6000.py" --root "$EXP" --scaling-root "$SCALING" \
 --ipc3-root "$IPC3" > "$EXP/logs/summary.log" 2>&1
rm -f "$EXP/status/running";date --iso-8601=seconds > "$EXP/status/complete"
