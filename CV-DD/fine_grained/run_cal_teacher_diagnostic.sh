#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: run_cal_teacher_diagnostic.sh DATASET}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1}"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-$EXP_ROOT/datasets}"

case "$DATASET" in
    CUB_imsize224|cub)
        DATASET=CUB_imsize224; ATTENTION_MAPS=8; CAL_RATIO=0.5; CAL_TEXT=5e-1 ;;
    A_imsize224|aircraft|a)
        DATASET=A_imsize224; ATTENTION_MAPS=32; CAL_RATIO=0.3; CAL_TEXT=3e-1 ;;
    SC_imsize224|cars|sc)
        DATASET=SC_imsize224; ATTENTION_MAPS=8; CAL_RATIO=0.3; CAL_TEXT=3e-1 ;;
    *) echo "Unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

DATA_DIR="$PREPARED_DATA_ROOT/$DATASET"
DIAGNOSTIC_ROOT="$EXP_ROOT/diagnostics/cal_teacher/$DATASET"
RAW_DIR="$DIAGNOSTIC_ROOT/raw"
EXTRACTED_DIR="$DIAGNOSTIC_ROOT/extracted"
EXP_NAME="ResNet18_M${ATTENTION_MAPS}_${CAL_TEXT}cal"
RAW_CHECKPOINT="$RAW_DIR/$EXP_NAME.pth"
LOG="$DIAGNOSTIC_ROOT/squeeze.log"
STATUS="$DIAGNOSTIC_ROOT/status.txt"
mkdir -p "$RAW_DIR" "$EXTRACTED_DIR"

if [[ -n "${WAIT_FOR_STATUS:-}" ]]; then
    WAIT_SECONDS="${WAIT_SECONDS:-300}"
    WAIT_STATUS_PATTERN="${WAIT_STATUS_PATTERN:-sampling complete}"
    while [[ ! -f "$WAIT_FOR_STATUS" ]] || \
          ! grep -q "$WAIT_STATUS_PATTERN" "$WAIT_FOR_STATUS"; do
        echo "$(date --iso-8601=seconds) waiting for status: $WAIT_FOR_STATUS"
        sleep "$WAIT_SECONDS"
    done
fi

[[ -d "$DATA_DIR/train" && -d "$DATA_DIR/test" ]] || {
    echo "Prepared dataset missing: $DATA_DIR" >&2
    exit 1
}

if [[ ! -f "$RAW_CHECKPOINT" ]]; then
    echo "$(date --iso-8601=seconds) training official joint CAL teacher" > "$STATUS"
    (
        cd "$ROOT_DIR/FD2"
        python -u squeeze/squeeze_cal.py \
            --dataset_name "$DATASET" --dataset_dir "$DATA_DIR" \
            --save_dir "$RAW_DIR" --model_list ResNet18 --model_source torchvision \
            --pretrained_weights --pretrained_bn --exp_name "$EXP_NAME" \
            --M "$ATTENTION_MAPS" --cal_ratio "$CAL_RATIO" --master_port 29620 \
            --epoch 160 --stop_epoch 50 --batch_size 4 --optimizer SGD \
            --world_size 1 --lr 1e-3 > "$LOG" 2>&1
    )
fi

python -u "$ROOT_DIR/CV-DD/fine_grained/extract_cal_backbone.py" \
    --dataset-name "$DATASET" --data-dir "$DATA_DIR" \
    --checkpoint "$RAW_CHECKPOINT" --output-dir "$EXTRACTED_DIR" \
    --attention-maps "$ATTENTION_MAPS" --cal-ratio "$CAL_RATIO" --workers 2
python "$ROOT_DIR/CV-DD/fine_grained/check_teacher_gate.py" \
    --dataset-name "$DATASET" --teacher-dir "$EXTRACTED_DIR"
echo "$(date --iso-8601=seconds) CAL teacher diagnostic complete" > "$STATUS"
