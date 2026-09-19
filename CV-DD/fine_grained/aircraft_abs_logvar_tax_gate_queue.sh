#!/usr/bin/env bash
set -euo pipefail
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
ROOT=/linxi/true_FG
OUT=/linxi/dataset/FGDD_metrics/aircraft_abs_logvar_tax_gate_v1
R0=/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard/selected/A_imsize224/rseed0/ipc3
mkdir -p "$OUT"/{logs,status}
cat > "$OUT/conditions.json" <<JSON
[
 {"name":"k2_f1","kind":"paired","path":"/tmp/fgdd_aircraft_six_source_fir_v1/fir_manifest.json","mode":"compressed_fir","tax_top1":0.3500350035003474},
 {"name":"k3_f1","kind":"paired","path":"/tmp/fgdd_k3_fir_v1/construction/three_source_fir_manifest.json","mode":"compressed_fir","tax_top1":0.7900790079007862},
 {"name":"k4_f1_v1","kind":"paired","path":"/tmp/fgdd_k4_fir_v1/construction/four_source_fir_manifest.json","mode":"compressed_fir","tax_top1":1.7001700170017056},
 {"name":"k4_f1_v2","kind":"paired","path":"/tmp/fgdd_k4_fir_v2/construction/four_source_fir_v2_manifest.json","mode":"compressed_fir","tax_top1":2.2902290229022904},
 {"name":"bbox","kind":"paired","path":"/tmp/fgdd_aircraft_six_source_experiment/construction/six_source_manifest.json","mode":"compressed","tax_top1":2.51025102510251},
 {"name":"object_decoded","kind":"paired","path":"/tmp/fgdd_aircraft_six_source_experiment/construction/six_source_manifest.json","mode":"object_decoded","tax_top1":1.8401840184018425},
 {"name":"plain","kind":"imagefolder","path":"/linxi/dataset/FGDD_NRR/aircraft_ipc3_pixel_optimization_v1/construction/selected/plain/ipc3","reference_root":"$R0","tax_top1":2.570257025702574},
 {"name":"cam","kind":"imagefolder","path":"/linxi/dataset/FGDD_NRR/aircraft_ipc3_pixel_optimization_v1/construction/selected/cam_protected/ipc3","reference_root":"$R0","tax_top1":0.9900990099009922},
 {"name":"random_protected","kind":"imagefolder","path":"/linxi/dataset/FGDD_NRR/aircraft_ipc3_pixel_optimization_v1/construction/selected/random_protected/ipc3","reference_root":"$R0","tax_top1":1.220122012201216}
]
JSON
students=(
 /linxi/dataset/FG_SoftAugFactorial_v2/aircraft_ipc3_v1/post_eval/A_imsize224/soft_fullframe_cutmix_random_real_1_s42/checkpoint.pth.tar
 /linxi/dataset/FG_SoftAugFactorial_v2/aircraft_ipc3_v1/post_eval/A_imsize224/soft_fullframe_cutmix_random_real_1_s43/checkpoint.pth.tar
 /linxi/dataset/FG_SoftAugFactorial_v2/aircraft_ipc3_v1/post_eval/A_imsize224/soft_fullframe_cutmix_random_real_1_s44/checkpoint.pth.tar
)
rm -f "$OUT/status/complete" "$OUT/status/failed"; date --iso-8601=seconds > "$OUT/status/running"
if CUDA_VISIBLE_DEVICES=0 python -u "$ROOT/CV-DD/fine_grained/audit_abs_logvar_tax_gate.py" \
 --conditions "$OUT/conditions.json" --student "${students[0]}" --student "${students[1]}" --student "${students[2]}" \
 --batch-size 16 --output "$OUT/summary.json" > "$OUT/logs/audit.log" 2>&1; then
 rm -f "$OUT/status/running"; date --iso-8601=seconds > "$OUT/status/complete"
else
 code=$?; rm -f "$OUT/status/running"; echo "$(date --iso-8601=seconds) exit=$code" > "$OUT/status/failed"; exit "$code"
fi
