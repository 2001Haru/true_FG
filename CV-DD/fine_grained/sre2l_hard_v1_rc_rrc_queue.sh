#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP="${EXP_ROOT:-/linxi/dataset/FG_HardLabel_standard/v1/aircraft_ipc3_sre2l_t42_r42_rc_rrc_v1}"
SOURCE_IMAGES=/linxi/dataset/FG_SRe2L_standard/v1/arms/tseed42/recovery/A_imsize224/rseed42/ipc3
IMAGES="$EXP/images/ipc3"
RECOVERY_MANIFEST=/linxi/dataset/FG_SRe2L_standard/v1/arms/tseed42/recovery/A_imsize224/rseed42/recovery_manifest.json
TEST=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/test
WEIGHTS=/linxi/models/torchvision/resnet18-f37072fd.pth
SOFT_REFERENCE=/linxi/dataset/FG_SRe2L_standard/v2/results/tseed42/A_imsize224/rseed42
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$EXP"/{logs,status,locks,results,checkpoints,summary,audits}
exec 9>"$EXP/locks/launcher.lock"; flock -n 9 || exit 75
trap 's=$?; if ((s)); then rm -f "$EXP/status/running"; echo "$(date --iso-8601=seconds) exit=$s" > "$EXP/status/failed"; fi' EXIT
rm -f "$EXP/status/failed" "$EXP/status/complete"; date --iso-8601=seconds > "$EXP/status/running"

python - "$SOURCE_IMAGES" "$IMAGES" "$RECOVERY_MANIFEST" "$EXP/audits/input.json" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
source,images,manifest,out=map(Path,sys.argv[1:])
x=json.loads(manifest.read_text())
assert x['status']=='complete' and x['dataset']['name']=='A_imsize224' and x['recovery_seed']==42 and x['dataset']['recovery_iterations']==4000
source_classes=sorted(p for p in source.iterdir() if p.is_dir())
assert [p.name for p in source_classes]==[f'new{i:03d}' for i in range(100)]
images.mkdir(parents=True,exist_ok=True)
for class_id,source_class in enumerate(source_classes):
 target=images/f'{class_id:03d}'
 if target.exists() or target.is_symlink():
  assert target.is_symlink() and target.resolve()==source_class.resolve()
 else:
  os.symlink(source_class.resolve(),target,target_is_directory=True)
files=sorted(source.glob('*/*.jpg')); assert len(files)==300
classes=sorted(p for p in images.iterdir() if p.is_dir()); assert [p.name for p in classes]==[f'{i:03d}' for i in range(100)]
assert all(len(list(p.glob('*.jpg')))==3 for p in classes)
h=hashlib.sha256()
for path in files:
 h.update(path.relative_to(source).as_posix().encode()); h.update(path.read_bytes())
payload={'status':'complete','dataset':'A_imsize224','ipc':3,'teacher_seed':42,'recovery_seed':42,
         'recovery_iterations':4000,'images':300,'classes':100,'image_tree_sha256':h.hexdigest(),
         'source_image_root':str(source.resolve()),'normalized_image_root':str(images.resolve()),
         'class_mapping':'000..099 directory symlinks to new000..new099; pixels unchanged',
         'recovery_manifest':str(manifest.resolve())}
out.parent.mkdir(parents=True,exist_ok=True); tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(payload,indent=2)+'\n');os.replace(tmp,out)
print(json.dumps(payload,indent=2))
PY

run_one(){
 local mode=$1 seed=$2 gpu=$3 result="$EXP/results/$mode/sseed$seed.json" protocol="hard_label_v1_${mode}" extra=()
 mkdir -p "$(dirname "$result")" "$EXP/checkpoints/$mode/sseed$seed"
 [[ $mode == rrc ]] && extra=(--train-crop-mode rrc --rrc-min-scale .08 --rrc-max-scale 1) || extra=(--train-crop-mode mild_rc)
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_hard_label_v1.py" --train-dir "$IMAGES" --val-dir "$TEST" \
  --dataset-name A_imsize224 --num-classes 100 --ipc 3 --student-seed "$seed" --result "$result" \
  --checkpoint-dir "$EXP/checkpoints/$mode/sseed$seed" --imagenet-weights-path "$WEIGHTS" --total-updates 3000 \
  --batch-size 64 --backbone-lr 3e-4 --head-lr 3e-3 --backbone-min-lr 0 --head-min-lr 0 --momentum .9 \
  --weight-decay 5e-4 --eval-every-updates 300 --workers 8 --persistent-workers --val-batch-size 256 \
  --protocol-name "$protocol" "${extra[@]}" > "$EXP/logs/eval_${mode}_s${seed}.log" 2>&1
 python "$ROOT/CV-DD/fine_grained/audit_hard_label_v1_result.py" --result "$result" --dataset A_imsize224 --classes 100 \
  --ipc 3 --student-seed "$seed" --validation-images 3333 --protocol-name "$protocol" --train-crop-mode "$mode" \
  --rrc-min-scale .08 --rrc-max-scale 1 >> "$EXP/logs/eval_${mode}_s${seed}.log" 2>&1
}
failed=0; pids=(); index=0
for mode in mild_rc rrc; do for seed in 42 43 44; do
 run_one "$mode" "$seed" $((index%2)) & pids+=("$!"); index=$((index+1))
 if ((${#pids[@]}==4)); then for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; pids=(); fi
done; done
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done; ((failed==0))
python "$ROOT/CV-DD/fine_grained/summarize_sre2l_hard_v1_rc_rrc.py" --root "$EXP" --soft-reference-root "$SOFT_REFERENCE" \
 > "$EXP/logs/summary.log" 2>&1
rm -f "$EXP/status/running"; date --iso-8601=seconds > "$EXP/status/complete"
