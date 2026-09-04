#!/usr/bin/env bash
set -euo pipefail

# This node ships an old ONNX-generated protobuf module beside a new protobuf
# runtime.  Torchvision imports ONNX transitively, so use the established
# pure-Python compatibility path used by the other fine-grained queues.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="${1:?usage: run_random_real_fg.sh prepare|eval DATASET IPC SELECTION_SEED [STUDENT_SEED]}"
DATASET="${2:?missing dataset}"
IPC="${3:?missing IPC}"
SELECTION_SEED="${4:?missing selection seed}"
STUDENT_SEED="${5:-}"

case "$DATASET" in
    CUB_imsize224|cub) DATASET=CUB_imsize224; CLASSES=200; VAL_IMAGES=5794; BATCH=20; ACCUM=2 ;;
    A_imsize224|aircraft|a) DATASET=A_imsize224; CLASSES=100; VAL_IMAGES=3333; BATCH=20; ACCUM=2 ;;
    SC_imsize224|cars|sc) DATASET=SC_imsize224; CLASSES=196; VAL_IMAGES=8041; BATCH=14; ACCUM=2 ;;
    *) echo "Unknown dataset: $DATASET" >&2; exit 2 ;;
esac
[[ "$IPC" =~ ^(1|3|5)$ ]] || { echo "IPC must be 1, 3, or 5" >&2; exit 2; }
[[ "$SELECTION_SEED" =~ ^(0|1|2)$ ]] || { echo "selection seed must be 0, 1, or 2" >&2; exit 2; }

DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
EXP_ROOT="${RANDOM_REAL_EXP_ROOT:-/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard}"
DATA_DIR="$DATA_ROOT/$DATASET"
SELECTED_DIR="$EXP_ROOT/selected/$DATASET/rseed${SELECTION_SEED}/ipc${IPC}"
MANIFEST="$EXP_ROOT/manifests/$DATASET/rseed${SELECTION_SEED}/ipc${IPC}.json"
RESULT="$EXP_ROOT/results/$DATASET/ipc${IPC}_rseed${SELECTION_SEED}_sseed${STUDENT_SEED}.json"
POST_ROOT="$EXP_ROOT/post_eval"

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
        mkdir -p "$(dirname "$RESULT")"
        exec 9>"${RESULT}.lock"
        flock -n 9 || fail "evaluation already running: $RESULT"
        if [[ ! -f "$RESULT" ]]; then
            python -u "$ROOT_DIR/CV-DD/validate/train_fkd.py" \
                --hard-label --model ResNet18 --ipc "$IPC" \
                --exp-name "random_real_${DATASET}_ipc${IPC}_rseed${SELECTION_SEED}_sseed${STUDENT_SEED}" \
                --original-data-path "$SELECTED_DIR" --output-dir "$POST_ROOT" \
                --batch-size "$BATCH" --epochs 400 --dataset-name "$DATASET" \
                --gradient-accumulation-steps "$ACCUM" --cos --workers 8 \
                --persistent-workers --fkd_seed 42 --train-seed "$STUDENT_SEED" \
                --student-initialization random --adamw-lr-override 1e-3 \
                --adamw-weight-decay 1e-5 --eta-override 2 --temperature 20 \
                --min-scale 0.08 --val-dir "$DATA_DIR/test" --disable-wandb \
                --per-class-output "$RESULT"
        fi
        python "$ROOT_DIR/CV-DD/fine_grained/annotate_random_real_result.py" \
            --result "$RESULT" --selection-manifest "$MANIFEST" \
            --selection-seed "$SELECTION_SEED"
        python "$ROOT_DIR/CV-DD/fine_grained/audit_result.py" \
            --result "$RESULT" --classes "$CLASSES" --validation-images "$VAL_IMAGES" \
            --expected-training-target hard_coarse_label
        ;;
    *) fail "unknown stage: $STAGE" ;;
esac
