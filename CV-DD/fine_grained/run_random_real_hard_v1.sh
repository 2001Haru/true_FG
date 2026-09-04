#!/usr/bin/env bash
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="${1:?usage: run_random_real_hard_v1.sh prepare|eval DATASET IPC SELECTION_SEED [STUDENT_SEED]}"
DATASET="${2:?missing dataset}"
IPC="${3:?missing IPC}"
SELECTION_SEED="${4:?missing selection seed}"
STUDENT_SEED="${5:-}"

case "$DATASET" in
    CUB_imsize224|cub) DATASET=CUB_imsize224; CLASSES=200; VAL_IMAGES=5794 ;;
    A_imsize224|aircraft|a) DATASET=A_imsize224; CLASSES=100; VAL_IMAGES=3333 ;;
    SC_imsize224|cars|sc) DATASET=SC_imsize224; CLASSES=196; VAL_IMAGES=8041 ;;
    *) echo "Unknown dataset: $DATASET" >&2; exit 2 ;;
esac
[[ "$IPC" =~ ^(1|3|5)$ ]] || { echo "IPC must be 1, 3, or 5" >&2; exit 2; }
[[ "$SELECTION_SEED" =~ ^(0|1|2)$ ]] || { echo "selection seed must be 0, 1, or 2" >&2; exit 2; }

DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
EXP_ROOT="${HARD_LABEL_V1_EXP_ROOT:-/linxi/dataset/FG_HardLabel_standard/v1/random_real}"
TORCHVISION_MODEL_ROOT="${TORCHVISION_MODEL_ROOT:-/linxi/models/torchvision}"
DATA_DIR="$DATA_ROOT/$DATASET"
SELECTED_DIR="$EXP_ROOT/selected/$DATASET/rseed${SELECTION_SEED}/ipc${IPC}"
MANIFEST="$EXP_ROOT/manifests/$DATASET/rseed${SELECTION_SEED}/ipc${IPC}.json"
RESULT="$EXP_ROOT/results/$DATASET/ipc${IPC}_rseed${SELECTION_SEED}_sseed${STUDENT_SEED}.json"
CHECKPOINT_DIR="$EXP_ROOT/checkpoints/$DATASET/ipc${IPC}_rseed${SELECTION_SEED}_sseed${STUDENT_SEED}"

fail() { echo "ERROR: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || fail "missing file: $1"; }

case "$STAGE" in
    prepare)
        python "$ROOT_DIR/CV-DD/fine_grained/prepare_random_real_fg.py" \
            --data-dir "$DATA_DIR" --output-dir "$SELECTED_DIR" --manifest "$MANIFEST" \
            --dataset-name "$DATASET" --classes "$CLASSES" --ipc "$IPC" \
            --selection-seed "$SELECTION_SEED"
        ;;
    eval)
        [[ "$STUDENT_SEED" =~ ^(42|43|44)$ ]] || fail "student seed must be 42, 43, or 44"
        require_file "$MANIFEST"
        mkdir -p "$(dirname "$RESULT")" "$CHECKPOINT_DIR"
        exec 9>"${RESULT}.lock"
        flock -n 9 || fail "evaluation already running: $RESULT"
        if [[ ! -f "$RESULT" ]]; then
            python -u "$ROOT_DIR/CV-DD/validate/train_hard_label_v1.py" \
                --train-dir "$SELECTED_DIR" --val-dir "$DATA_DIR/test" \
                --dataset-name "$DATASET" --num-classes "$CLASSES" --ipc "$IPC" \
                --student-seed "$STUDENT_SEED" --result "$RESULT" \
                --checkpoint-dir "$CHECKPOINT_DIR" \
                --imagenet-weights-path "$TORCHVISION_MODEL_ROOT/resnet18-f37072fd.pth" \
                --total-updates 3000 \
                --batch-size 64 --backbone-lr 3e-4 --head-lr 3e-3 \
                --backbone-min-lr 0 --head-min-lr 0 --momentum 0.9 \
                --weight-decay 5e-4 --eval-every-updates 300 \
                --workers 8 --persistent-workers --val-batch-size 256
        fi
        python "$ROOT_DIR/CV-DD/fine_grained/annotate_random_real_result.py" \
            --result "$RESULT" --selection-manifest "$MANIFEST" \
            --selection-seed "$SELECTION_SEED"
        python "$ROOT_DIR/CV-DD/fine_grained/audit_hard_label_v1_result.py" \
            --result "$RESULT" --dataset "$DATASET" --classes "$CLASSES" \
            --ipc "$IPC" --student-seed "$STUDENT_SEED" \
            --validation-images "$VAL_IMAGES"
        ;;
    *) fail "unknown stage: $STAGE" ;;
esac
