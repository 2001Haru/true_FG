#!/usr/bin/env bash
set -euo pipefail
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${OUT:-/linxi/dataset/FGDD_metrics/aircraft_cmmd_nonzero_tax_gate_v1}"
mkdir -p "$OUT/logs" "$OUT/status"
cat > "$OUT/conditions.json" <<'JSON'
[
 {"name":"k2_f1","kind":"paired","path":"/tmp/fgdd_aircraft_six_source_fir_v1/fir_manifest.json","mode":"compressed_fir","tax_top1":0.3500350035003474},
 {"name":"k3_f1","kind":"paired","path":"/tmp/fgdd_k3_fir_v1/construction/three_source_fir_manifest.json","mode":"compressed_fir","tax_top1":0.7900790079007862},
 {"name":"k4_f1_v1","kind":"paired","path":"/tmp/fgdd_k4_fir_v1/construction/four_source_fir_manifest.json","mode":"compressed_fir","tax_top1":1.7001700170017056},
 {"name":"k4_f1_v2","kind":"paired","path":"/tmp/fgdd_k4_fir_v2/construction/four_source_fir_v2_manifest.json","mode":"compressed_fir","tax_top1":2.2902290229022904},
 {"name":"bbox","kind":"paired","path":"/tmp/fgdd_aircraft_six_source_experiment/construction/six_source_manifest.json","mode":"compressed","tax_top1":2.51025102510251},
 {"name":"object_decoded","kind":"paired","path":"/tmp/fgdd_aircraft_six_source_experiment/construction/six_source_manifest.json","mode":"object_decoded","tax_top1":1.8401840184018425},
 {"name":"plain","kind":"imagefolder","path":"/linxi/dataset/FGDD_NRR/aircraft_ipc3_pixel_optimization_v1/construction/selected/plain/ipc3","tax_top1":2.570257025702574},
 {"name":"cam","kind":"imagefolder","path":"/linxi/dataset/FGDD_NRR/aircraft_ipc3_pixel_optimization_v1/construction/selected/cam_protected/ipc3","tax_top1":0.9900990099009922},
 {"name":"random_protected","kind":"imagefolder","path":"/linxi/dataset/FGDD_NRR/aircraft_ipc3_pixel_optimization_v1/construction/selected/random_protected/ipc3","tax_top1":1.220122012201216}
]
JSON
rm -f "$OUT/status/complete" "$OUT/status/failed"
echo "$(date --iso-8601=seconds) started" > "$OUT/status/running"
if CUDA_VISIBLE_DEVICES=0 python -u "$ROOT/CV-DD/fine_grained/audit_cmmd_tax_gate.py" \
 --conditions "$OUT/conditions.json" \
 --geometry-fkd /linxi/dataset/FG_HardAugFactorial_v2/aircraft_ipc3_v1/fkd/rrc_no_cutmix/random_real/source_seed0/ipc3_bs20_ipc3 \
 --reference-features /linxi/dataset/FGDD_metrics/aircraft_f1_k_cmmd_vendi_v1/features \
 --clip-source-root "$ROOT/third_party/metric_backbones/CLIP" --output-root "$OUT" \
 --batch-size 32 --workers 8 > "$OUT/logs/audit.log" 2>&1; then
 rm -f "$OUT/status/running"; echo "$(date --iso-8601=seconds) complete" > "$OUT/status/complete"
else
 code=$?; rm -f "$OUT/status/running"; echo "$(date --iso-8601=seconds) exit=$code" > "$OUT/status/failed"; exit "$code"
fi
