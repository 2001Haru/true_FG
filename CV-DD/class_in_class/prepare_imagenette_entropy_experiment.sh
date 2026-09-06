#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${IMAGENETTE_DATA_ROOT:-/linxi/dataset/CV-DD/test_data/imagenet-nette}"
TEACHER="${IMAGENETTE_C1_TEACHER:-/linxi/dataset/CV-DD/offline_models/imagenet-nette/ResNet18.pth}"
DINO_MODEL="${DINOV2_MODEL_ROOT:-/linxi/models/DINOv2/dinov2-base}"
EXP_ROOT="${IMAGENETTE_ENTROPY_ROOT:-/linxi/dataset/CV-DD/experiments/imagenette_entropy_selection_v1}"
DINO_CACHE="$EXP_ROOT/cache/imagenette_dinov2.pt"
mkdir -p "$EXP_ROOT/cache" "$EXP_ROOT/logs/preflight"

[[ -d "$DATA_ROOT/train" && -d "$DATA_ROOT/test" ]] || { echo "missing ImageNette official split" >&2; exit 1; }
[[ -f "$TEACHER" ]] || { echo "missing fixed C1 Teacher: $TEACHER" >&2; exit 1; }
[[ -d "$DINO_MODEL" ]] || { echo "missing DINOv2 model: $DINO_MODEL" >&2; exit 1; }

if [[ ! -f "$DINO_CACHE" ]]; then
    CUDA_VISIBLE_DEVICES="${PREFLIGHT_GPU:-0}" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    python -u "$ROOT/class_in_class/encode_imagenette_dinov2.py" \
        --data-root "$DATA_ROOT" --model-dir "$DINO_MODEL" --output "$DINO_CACHE" \
        --splits train test --batch-size 128 --workers "${PREFLIGHT_WORKERS:-16}" \
        >"$EXP_ROOT/logs/preflight/dino.log" 2>&1
fi
CUDA_VISIBLE_DEVICES="${PREFLIGHT_GPU:-0}" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
python -u "$ROOT/class_in_class/prepare_imagenette_entropy_selection.py" \
    --data-root "$DATA_ROOT" --teacher-checkpoint "$TEACHER" \
    --dino-cache "$DINO_CACHE" --output-root "$EXP_ROOT" \
    --batch-size 256 --workers "${PREFLIGHT_WORKERS:-16}" \
    >"$EXP_ROOT/logs/preflight/selection.log" 2>&1

echo "Preflight complete. Inspect $EXP_ROOT/preflight/selection_preflight.json and contact_sheets/."
echo "Training remains blocked until manual_visual_gate.json is written with approved=true."

