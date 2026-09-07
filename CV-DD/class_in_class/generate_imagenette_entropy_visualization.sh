#!/usr/bin/env bash
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-20}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-20}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${IMAGENETTE_ENTROPY_ROOT:-/linxi/dataset/CV-DD/experiments/imagenette_entropy_selection_v1}"
DATA_ROOT="${IMAGENETTE_DATA_ROOT:-/linxi/dataset/CV-DD/test_data/imagenet-nette}"
TEACHER="${IMAGENETTE_C1_TEACHER:-/linxi/dataset/CV-DD/offline_models/imagenet-nette/ResNet18.pth}"
DINO_CACHE="$EXP_ROOT/cache/imagenette_dinov2.pt"
PER_IMAGE="$EXP_ROOT/preflight/per_image_teacher_entropy_and_dino.json"
PREDICTION_CACHE="$EXP_ROOT/cache/imagenette_calibration_mean_probabilities.pt"
OUTPUT_DIR="$EXP_ROOT/figures/entropy_feature_maps"
LOG_DIR="$EXP_ROOT/logs/entropy_feature_maps"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

for path in "$DATA_ROOT/train" "$TEACHER" "$DINO_CACHE" "$PER_IMAGE"; do
  [[ -e "$path" ]] || { echo "missing visualization input: $path" >&2; exit 1; }
done
if [[ ! -f "$PREDICTION_CACHE" ]]; then
  CUDA_VISIBLE_DEVICES="${VIS_GPU:-0}" python -u "$ROOT/class_in_class/extract_imagenette_calibration_predictions.py" \
    --data-root "$DATA_ROOT" --teacher-checkpoint "$TEACHER" \
    --per-image-audit "$PER_IMAGE" --output "$PREDICTION_CACHE" \
    --batch-size "${VIS_BATCH_SIZE:-512}" --workers "${VIS_WORKERS:-16}" \
    >"$LOG_DIR/extract_predictions.log" 2>&1
fi
python -u "$ROOT/class_in_class/plot_imagenette_entropy_feature_maps.py" \
  --dino-cache "$DINO_CACHE" --prediction-cache "$PREDICTION_CACHE" \
  --per-image-audit "$PER_IMAGE" --output-dir "$OUTPUT_DIR" \
  >"$LOG_DIR/plot.log" 2>&1
echo "complete: $OUTPUT_DIR"
