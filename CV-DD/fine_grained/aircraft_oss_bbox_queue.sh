#!/usr/bin/env bash
set -euo pipefail
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
DEPS_ROOT="${DEPS_ROOT:-/tmp/fg_oss_bbox_deps}"
export PYTHONPATH="$DEPS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
GD="$ROOT/third_party/detectors/GroundingDINO"
SAM="$ROOT/third_party/segmenters/sam2"
MODEL_ROOT="${MODEL_ROOT:-/linxi/models/oss_bbox}"
OUT="${OUT:-/linxi/dataset/FGDD_bbox_models/aircraft_oss_v1}"
RAW=/linxi/dataset/FD2/raw/fgvc-aircraft-2013b/data
GD_CKPT="$MODEL_ROOT/groundingdino_swint_ogc.pth"
SAM_CKPT="$MODEL_ROOT/sam2.1_hiera_large.pt"
BERT="$MODEL_ROOT/bert-base-uncased"
mkdir -p "$OUT"/{logs,status} "$MODEL_ROOT"
exec 9>"$OUT/launcher.lock"; flock -n 9 || exit 75
rm -f "$OUT/status/complete" "$OUT/status/failed"; date --iso-8601=seconds > "$OUT/status/running"
trap 's=$?; rm -f "$OUT/status/running"; if ((s)); then echo "$(date --iso-8601=seconds) exit=$s" > "$OUT/status/failed"; fi' EXIT

[[ -f "$GD_CKPT" ]] || curl -fL --retry 5 -o "$GD_CKPT.tmp" \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
[[ -f "$GD_CKPT" ]] || mv "$GD_CKPT.tmp" "$GD_CKPT"
[[ -f "$SAM_CKPT" ]] || curl -fL --retry 5 -o "$SAM_CKPT.tmp" \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
[[ -f "$SAM_CKPT" ]] || mv "$SAM_CKPT.tmp" "$SAM_CKPT"

"$PYTHON" -u "$ROOT/CV-DD/fine_grained/audit_aircraft_oss_boxes.py" \
  --raw-images "$RAW/images" --boxes "$RAW/images_box.txt" --variants "$RAW/images_variant_trainval.txt" \
  --grounding-root "$GD" --grounding-config "$GD/groundingdino/config/GroundingDINO_SwinT_OGC.py" \
  --grounding-checkpoint "$GD_CKPT" --text-encoder-root "$BERT" --sam-root "$SAM" \
  --sam-config configs/sam2.1/sam2.1_hiera_l.yaml --sam-checkpoint "$SAM_CKPT" \
  --caption 'airplane.' --device cuda --output-root "$OUT" > "$OUT/logs/audit.log" 2>&1
rm -f "$OUT/status/running"; date --iso-8601=seconds > "$OUT/status/complete"
