#!/usr/bin/env bash
set -euo pipefail
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCHIVE=/linxi/dataset/FGDD_six_source/aircraft_ipc3_bbox_pack_v1_complete.tar.gz
WORK=/tmp/fgdd_aircraft_six_source_experiment
FKD_WORK=/tmp/fgdd_u6_bbox_bn_cross_v1/compressed_fkd
OUT=/linxi/dataset/FGDD_BN_audits/aircraft_u6_bbox_cross_bn_v1
TEST=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/test
mkdir -p "$OUT"/{logs,status} "$FKD_WORK"
exec 9>"$OUT/launcher.lock"; flock -n 9 || exit 75
rm -f "$OUT/status/complete" "$OUT/status/failed"
date --iso-8601=seconds > "$OUT/status/running"
trap 's=$?; rm -f "$OUT/status/running"; if ((s)); then echo "$(date --iso-8601=seconds) exit=$s" > "$OUT/status/failed"; fi' EXIT

# The original experiment was archived from /tmp.  Restore only the paired
# construction, U6 checkpoints, and compressed-view FKD archive at the exact
# paths embedded in its manifest.
if [[ ! -f "$WORK/construction/six_source_manifest.json" ]]; then
  tar -xzf "$ARCHIVE" -C /tmp \
    fgdd_aircraft_six_source_experiment/construction \
    fgdd_aircraft_six_source_experiment/post_eval/soft/reference \
    fgdd_aircraft_six_source_experiment/fkd_archives/compressed.tar
fi
if [[ ! -f "$FKD_WORK/relabel_manifest.json" ]]; then
  tar -xf "$WORK/fkd_archives/compressed.tar" -C "$FKD_WORK"
fi

python - "$WORK" "$FKD_WORK" "$OUT/preflight.json" <<'PY'
import glob,json,sys,torch
from pathlib import Path
work,fkd,out=map(Path,sys.argv[1:])
manifest=json.load(open(work/'construction/six_source_manifest.json'))
checkpoints=sorted(glob.glob(str(work/'post_eval/soft/reference/sseed*/A_imsize224/six_source_soft_reference_s*/checkpoint.pth.tar')))
files=list(fkd.glob('epoch_*/batch_*.tar'))
assert manifest['status']=='complete' and manifest['parents']==300 and manifest['sources_per_parent']==2
assert len(checkpoints)==3 and len(files)==6000
for path in checkpoints:
    payload=torch.load(path,map_location='cpu',weights_only=False)
    assert 'state_dict' in payload
out.write_text(json.dumps({'status':'complete','checkpoints':checkpoints,'fkd_batches':len(files),
 'manifest':str((work/'construction/six_source_manifest.json').resolve()),
 'recalibration_domain':'BBox decoded images with compressed-arm FKD flip/CutMix trajectory, epochs0-39',
 'fixed_model':'U6 reference-image Student final checkpoints'},indent=2)+'\n')
PY

CUDA_VISIBLE_DEVICES=0 python -u "$ROOT/CV-DD/fine_grained/audit_paired_same_domain_bn.py" \
  --manifest "$WORK/construction/six_source_manifest.json" --mode compressed --fkd "$FKD_WORK" \
  --checkpoints "$WORK/post_eval/soft/reference/sseed*/A_imsize224/six_source_soft_reference_s*/checkpoint.pth.tar" \
  --test-root "$TEST" --epochs 40 --device cuda:0 --output "$OUT/summary.json" \
  > "$OUT/logs/audit.log" 2>&1

python - "$OUT/summary.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); assert x['status']=='complete' and len(x['rows'])==3
assert abs(x['summary']['before_top1']['mean']-76.0976)<.01
assert max(row['parameter_max_abs_change'] for row in x['rows'])==0
PY
rm -f "$OUT/status/running"; date --iso-8601=seconds > "$OUT/status/complete"
