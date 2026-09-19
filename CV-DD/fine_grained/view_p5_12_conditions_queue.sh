#!/usr/bin/env bash
set -euo pipefail
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ROOT=/linxi/true_FG/CV-DD/fine_grained
OUT=/linxi/dataset/FGDD_BN_audits/aircraft_view_p5_12_v1
SAMPLED_EPOCHS=${SAMPLED_EPOCHS:-40}
mkdir -p "$OUT"/{logs,status}
cat >"$OUT/conditions.json" <<'JSON'
{"conditions":[
 {"name":"r0","kind":"zero","tax_top1":0.590059005900585},
 {"name":"kmeans","kind":"zero","tax_top1":0.5600560056005625},
 {"name":"u6","kind":"zero","tax_top1":0.7800780078007818},
 {"name":"k2_f1","kind":"paired","manifest":"/tmp/fgdd_aircraft_six_source_fir_v1/fir_manifest.json","mode":"compressed_fir","fkd":"/tmp/fgdd_aircraft_six_source_fir_v1/fkd","tax_top1":0.3500350035003474},
 {"name":"k3_f1","kind":"paired","manifest":"/tmp/fgdd_k3_fir_v1/construction/three_source_fir_manifest.json","mode":"compressed_fir","fkd":"/tmp/fgdd_k3_fir_v1/fkd/fir","tax_top1":0.7900790079007862},
 {"name":"k4_f1_v1","kind":"paired","manifest":"/tmp/fgdd_k4_fir_v1/construction/four_source_fir_manifest.json","mode":"compressed_fir","fkd":"/tmp/fgdd_k4_fir_v1/fkd/fir","tax_top1":1.7001700170017056},
 {"name":"k4_f1_v2","kind":"paired","manifest":"/tmp/fgdd_k4_fir_v2/construction/four_source_fir_v2_manifest.json","mode":"compressed_fir","fkd":"/tmp/fgdd_k4_fir_v2/fkd","tax_top1":2.2902290229022904},
 {"name":"bbox","kind":"paired","manifest":"/tmp/fgdd_aircraft_six_source_experiment/construction/six_source_manifest.json","mode":"compressed","fkd":"/tmp/fgdd_aircraft_six_source_stage/fkd/compressed","tax_top1":2.51025102510251},
 {"name":"object_decoded","kind":"paired","manifest":"/tmp/fgdd_aircraft_six_source_experiment/construction/six_source_manifest.json","mode":"object_decoded","fkd":"/tmp/fgdd_aircraft_six_source_loss_decomposition/fkd/object_decoded","tax_top1":1.8401840184018425},
 {"name":"plain","kind":"imagefolder","path":"/linxi/dataset/FGDD_NRR/aircraft_ipc3_pixel_optimization_v1/construction/selected/plain/ipc3","reference_root":"/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard/selected/A_imsize224/rseed0/ipc3","fkd":"/linxi/dataset/FGDD_NRR/aircraft_ipc3_pixel_optimization_v1/fkd/plain","tax_top1":2.570257025702574},
 {"name":"cam","kind":"imagefolder","path":"/linxi/dataset/FGDD_NRR/aircraft_ipc3_pixel_optimization_v1/construction/selected/cam_protected/ipc3","reference_root":"/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard/selected/A_imsize224/rseed0/ipc3","fkd":"/linxi/dataset/FGDD_NRR/aircraft_ipc3_pixel_optimization_v1/fkd/cam_protected","tax_top1":0.9900990099009922},
 {"name":"random_protected","kind":"imagefolder","path":"/linxi/dataset/FGDD_NRR/aircraft_ipc3_pixel_optimization_v1/construction/selected/random_protected/ipc3","reference_root":"/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard/selected/A_imsize224/rseed0/ipc3","fkd":"/linxi/dataset/FGDD_NRR/aircraft_ipc3_pixel_optimization_v1/fkd/random_protected","tax_top1":1.220122012201216}
]}
JSON
STUDENTS=(
 /linxi/dataset/FG_SoftAugFactorial_v2/aircraft_ipc3_v1/post_eval/A_imsize224/soft_fullframe_cutmix_random_real_1_s42/checkpoint.pth.tar
 /linxi/dataset/FG_SoftAugFactorial_v2/aircraft_ipc3_v1/post_eval/A_imsize224/soft_fullframe_cutmix_random_real_1_s43/checkpoint.pth.tar
 /linxi/dataset/FG_SoftAugFactorial_v2/aircraft_ipc3_v1/post_eval/A_imsize224/soft_fullframe_cutmix_random_real_1_s44/checkpoint.pth.tar
)
date --iso-8601=seconds >"$OUT/status/running"
CUDA_VISIBLE_DEVICES=0 python "$ROOT/audit_view_p5_12_conditions.py" --conditions "$OUT/conditions.json" \
 --student "${STUDENTS[0]}" --student "${STUDENTS[1]}" --student "${STUDENTS[2]}" \
 --sampled-epochs "$SAMPLED_EPOCHS" --output "$OUT/view_p5.json" >"$OUT/logs/view_p5.log" 2>&1
rm -f "$OUT/status/running";date --iso-8601=seconds >"$OUT/status/complete"
