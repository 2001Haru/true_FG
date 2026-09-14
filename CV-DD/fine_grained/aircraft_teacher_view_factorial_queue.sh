#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_ViT_Teachers/aircraft_seed42_224_v1}"
OUT="$EXP_ROOT/audits/view_factorial"
LOG="$EXP_ROOT/logs/view_factorial"
IMAGES=/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard/selected/A_imsize224/rseed0/ipc3
RRC=/linxi/dataset/FG_HardAugFactorial_v2/aircraft_ipc3_v1/fkd/rrc_no_cutmix/random_real/source_seed0/ipc3_bs20_ipc3
CUTMIX=/linxi/dataset/FG_HardAugFactorial_v2/aircraft_ipc3_v1/fkd/fullframe_cutmix/random_real/source_seed0/ipc3_bs20_ipc3
RESNET=/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42/ResNet18.pth
EPOCHS=(0 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 340 360 380 399)

mkdir -p "$OUT" "$LOG" "$EXP_ROOT/status" "$EXP_ROOT/locks"

run_one(){
  local labeler="$1" gpu="$2" checkpoint="$3" source="${4:-}"
  local source_args=()
  if [[ -n "$source" ]]; then source_args=(--source-root "$source"); fi
  CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -u "$ROOT_DIR/CV-DD/fine_grained/audit_teacher_view_factorial.py" \
    --labeler "$labeler" --checkpoint "$checkpoint" "${source_args[@]}" \
    --images "$IMAGES" --rrc-fkd "$RRC" --cutmix-fkd "$CUTMIX" \
    --sample-epochs "${EPOCHS[@]}" --batch-size 20 --seed 42 \
    --output "$OUT/$labeler.json" > "$LOG/$labeler.log" 2>&1
}

main(){
  exec 9>"$EXP_ROOT/locks/view_factorial.lock"
  flock -n 9 || exit 75
  rm -f "$EXP_ROOT/status/view_factorial.failed" "$EXP_ROOT/status/view_factorial.complete"
  date --iso-8601=seconds > "$EXP_ROOT/status/view_factorial.running"
  run_one resnet_bssl 0 "$RESNET" & p0=$!
  run_one resnet_eval 1 "$RESNET" & p1=$!
  run_one vit 0 "$EXP_ROOT/teachers/vit/final_step10000.pth" "$ROOT_DIR/third_party/teacher_backbones/ViT-pytorch" & p2=$!
  run_one transfg 1 "$EXP_ROOT/teachers/transfg/final_step10000.pth" "$ROOT_DIR/third_party/teacher_backbones/TransFG" & p3=$!
  failed=0
  for pid in "$p0" "$p1" "$p2" "$p3"; do wait "$pid" || failed=1; done
  (( failed == 0 ))
  python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
for name in ('resnet_bssl','resnet_eval','vit','transfg'):
    x=json.load(open(root/f'{name}.json'))
    assert x['status']=='complete' and x['views_per_cell']==6300, (name,x.get('status'),x.get('views_per_cell'))
    assert x['metadata_checks']['full_coords_nonfull']==0
    assert x['metadata_checks']['unexpected_rrc_mix']==0
print('view factorial completion gate passed')
PY
  rm -f "$EXP_ROOT/status/view_factorial.running"
  date --iso-8601=seconds > "$EXP_ROOT/status/view_factorial.complete"
}

trap 'code=$?; if (( code != 0 )); then rm -f "$EXP_ROOT/status/view_factorial.running"; echo "$(date --iso-8601=seconds) exit=$code" > "$EXP_ROOT/status/view_factorial.failed"; fi' EXIT
main
