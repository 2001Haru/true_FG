#!/usr/bin/env bash
set -euo pipefail
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ARCHIVE=/linxi/dataset/FGDD_six_source/aircraft_ipc3_bbox_pack_v1_complete.tar.gz
WORK=/tmp/fgdd_aircraft_six_source_experiment
F1_WORK=/tmp/fgdd_aircraft_six_source_fir_v1
FKD_WORK=/tmp/fgdd_u6_bbox_bn_cross_v1/compressed_fkd
FIT=/linxi/dataset/FGDD_six_source/fir_preemphasis_v1/audits/fit.json
OUT="${OUT:-/linxi/dataset/FGDD_BN_audits/aircraft_bbox_model_f1_bn_v1}"
TEST=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/test
mkdir -p "$OUT"/{logs,status} "$F1_WORK" "$FKD_WORK"
exec 9>"$OUT/launcher.lock"; flock -n 9 || exit 75
rm -f "$OUT/status/complete" "$OUT/status/failed"; date --iso-8601=seconds > "$OUT/status/running"
trap 's=$?; rm -f "$OUT/status/running"; if ((s)); then echo "$(date --iso-8601=seconds) exit=$s" > "$OUT/status/failed"; fi' EXIT

if [[ ! -f "$WORK/construction/six_source_manifest.json" || ! -d "$WORK/post_eval/soft/compressed" ]]; then
  tar -xzf "$ARCHIVE" -C /tmp \
    fgdd_aircraft_six_source_experiment/construction \
    fgdd_aircraft_six_source_experiment/post_eval/soft/compressed \
    fgdd_aircraft_six_source_experiment/fkd_archives/compressed.tar
fi
if [[ ! -f "$FKD_WORK/relabel_manifest.json" ]]; then tar -xf "$WORK/fkd_archives/compressed.tar" -C "$FKD_WORK"; fi

# Reconstruct the original F1 manifest exactly: the image/source records are
# the frozen BBox manifest and the fitted decoder is the archived fit config.
python - "$WORK/construction/six_source_manifest.json" "$FIT" "$F1_WORK/fir_manifest.json" "$OUT/preflight.json" <<'PY'
import glob,json,sys
from pathlib import Path
base_path,fit_path,out_path,audit_path=map(Path,sys.argv[1:])
base=json.load(open(base_path));fit=json.load(open(fit_path));configuration=fit['configuration']
assert fit['status']=='complete' and configuration['protocol']=='aircraft_bbox6_fir_preemphasis_v1'
base['fir_preemphasis']=configuration;base['protocol']=base['protocol']+'+fir_preemphasis_v1'
out_path.write_text(json.dumps(base,indent=2)+'\n')
checkpoints=sorted(glob.glob('/tmp/fgdd_aircraft_six_source_experiment/post_eval/soft/compressed/sseed*/A_imsize224/six_source_soft_compressed_s*/checkpoint.pth.tar'))
assert len(checkpoints)==3 and base['parents']==300 and base['sources_per_parent']==2
audit_path.write_text(json.dumps({'status':'complete','fixed_model':'BBox-6 final Student checkpoints',
 'calibration_pixels':'same stored sources decoded with archived F1-v1 FIR','checkpoints':checkpoints,
 'geometry':'BBox compressed-arm FKD epochs0-39; identical parent/source/flip/CutMix schedule',
 'parameter_policy':'all parameters requires_grad=False; model eval; only BatchNorm2d modules set train'},indent=2)+'\n')
PY

CUDA_VISIBLE_DEVICES=0 python -u "$ROOT/CV-DD/fine_grained/audit_paired_same_domain_bn.py" \
  --manifest "$F1_WORK/fir_manifest.json" --mode compressed_fir --fkd "$FKD_WORK" \
  --checkpoints "$WORK/post_eval/soft/compressed/sseed*/A_imsize224/six_source_soft_compressed_s*/checkpoint.pth.tar" \
  --test-root "$TEST" --epochs 40 --device cuda:0 --output "$OUT/summary.json" > "$OUT/logs/audit.log" 2>&1
python - "$OUT/summary.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));assert x['status']=='complete' and len(x['rows'])==3
assert abs(x['summary']['before_top1']['mean']-73.8074)<.01
assert max(row['parameter_max_abs_change'] for row in x['rows'])==0
PY
rm -f "$OUT/status/running"; date --iso-8601=seconds > "$OUT/status/complete"
