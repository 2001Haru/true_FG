#!/usr/bin/env bash
set -euo pipefail
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-/linxi/dataset/FGDD_metrics/aircraft_cmmd_common_views_v1}"
SCRIPT="$ROOT_DIR/CV-DD/fine_grained/audit_cmmd_common_views.py"
GEOM=/linxi/dataset/FG_HardAugFactorial_v2/aircraft_ipc3_v1/fkd/rrc_no_cutmix/random_real/source_seed0/ipc3_bs20_ipc3
REF=/linxi/dataset/FGDD_metrics/aircraft_f1_k_cmmd_vendi_v1/features
TEACHER=/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42/ResNet18.pth
CLIP="$ROOT_DIR/third_party/metric_backbones/CLIP"
R0=/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard/selected/A_imsize224/rseed0/ipc3
RDED=/linxi/dataset/FG_RDED_original_v2/aircraft_v1/generated/A_imsize224/gseed42/ipc3
K2=/tmp/fgdd_aircraft_six_source_fir_v1/fir_manifest.json
K3=/tmp/fgdd_k3_fir_v1/construction/three_source_fir_manifest.json
K4=/tmp/fgdd_k4_fir_v1/construction/four_source_fir_manifest.json
K4V2=/tmp/fgdd_k4_fir_v2/construction/four_source_fir_v2_manifest.json
mkdir -p "$OUT/logs" "$OUT/status"
common=(--geometry-fkd "$GEOM" --reference-features "$REF" --teacher "$TEACHER"
 --clip-source-root "$CLIP" --output-root "$OUT" --views 32 --batch-size 32 --workers 8
 --image-spec "r0=$R0" --image-spec "rded=$RDED"
 --paired-spec "k2_f1_v1=$K2" --paired-spec "k3_f1_v1=$K3"
 --paired-spec "k4_f1_v1=$K4" --paired-spec "k4_f1_v2=$K4V2")
rm -f "$OUT/status/complete" "$OUT/status/failed"
echo "$(date --iso-8601=seconds) starting common CutMix-off N9600/N600 views" > "$OUT/status/running"
set +e
CUDA_VISIBLE_DEVICES=0 python -u "$SCRIPT" --policy off "${common[@]}" > "$OUT/logs/off.log" 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES=1 python -u "$SCRIPT" --policy on  "${common[@]}" > "$OUT/logs/on.log"  2>&1 & p1=$!
wait "$p0"; s0=$?; wait "$p1"; s1=$?
set -e
rm -f "$OUT/status/running"
if ((s0 || s1)); then echo "$(date --iso-8601=seconds) off=$s0 on=$s1" > "$OUT/status/failed"; exit 1; fi
python - "$OUT" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);off=json.load(open(r/'summary_off.json'));on=json.load(open(r/'summary_on.json'))
assert off['geometry_schedule_sha256']==on['geometry_schedule_sha256']
out={'status':'complete','protocol':'aircraft_cmmd_common_views_v1','off':off,'on':on}
(r/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
PY
echo "$(date --iso-8601=seconds) complete" > "$OUT/status/complete"
