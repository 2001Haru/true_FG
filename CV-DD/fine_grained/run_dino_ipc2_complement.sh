#!/usr/bin/env bash
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="${1:?usage: run_dino_ipc2_complement.sh prepare|eval DATASET [ARM] [STUDENT_SEED]}"
DATASET="${2:?missing dataset}"
ARM="${3:-}"
STUDENT_SEED="${4:-}"

case "$DATASET" in
    CUB_imsize224|cub) DATASET=CUB_imsize224; CLASSES=200; VAL_IMAGES=5794 ;;
    A_imsize224|aircraft|a) DATASET=A_imsize224; CLASSES=100; VAL_IMAGES=3333 ;;
    SC_imsize224|cars|sc) DATASET=SC_imsize224; CLASSES=196; VAL_IMAGES=8041 ;;
    *) echo "Unknown dataset: $DATASET" >&2; exit 2 ;;
esac

DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
TORCHVISION_MODEL_ROOT="${TORCHVISION_MODEL_ROOT:-/linxi/models/torchvision}"
PARENT_ROOT="${DINO_IPC1_PARENT_ROOT:-/linxi/dataset/FG_HardLabel_standard/v1/dino_fivearm_ipc1}"
EXP_ROOT="${DINO_IPC2_EXP_ROOT:-/linxi/dataset/FG_HardLabel_standard/v1/dino_ipc2_center_complement}"
DATA_DIR="$DATA_ROOT/$DATASET"
SELECTION_AUDIT="$EXP_ROOT/selection_audits/$DATASET.json"

fail() { echo "ERROR: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || fail "missing file: $1"; }

case "$STAGE" in
    prepare)
        python "$ROOT_DIR/CV-DD/fine_grained/prepare_dino_ipc2_complement.py" \
            --parent-experiment-root "$PARENT_ROOT" --output-root "$EXP_ROOT" \
            --audit-output "$SELECTION_AUDIT" --dataset-name "$DATASET" \
            --classes "$CLASSES"
        ;;
    eval)
        case "$ARM" in
            random_ipc2_rseed[012]|spherical_kmeans2_rseed[012]|center_plus_random_rseed[012]|center_plus_shell_random_rseed[012])
                [[ "$STUDENT_SEED" =~ ^(42|43|44)$ ]] \
                    || fail "stochastic arm requires Student seed 42..44"
                ;;
            center_plus_outward|center_plus_high_margin|center_plus_rival_facing|global_center_top2)
                [[ "$STUDENT_SEED" =~ ^(42|43|44|45|46|47)$ ]] \
                    || fail "deterministic arm requires Student seed 42..47"
                ;;
            *) fail "unknown IPC2 arm: $ARM" ;;
        esac
        MANIFEST="$EXP_ROOT/manifests/$DATASET/$ARM.json"
        RESULT="$EXP_ROOT/results/$DATASET/$ARM/sseed${STUDENT_SEED}.json"
        CHECKPOINT_DIR="$EXP_ROOT/checkpoints/$DATASET/$ARM/sseed${STUDENT_SEED}"
        require_file "$MANIFEST"
        require_file "$SELECTION_AUDIT"
        mkdir -p "$(dirname "$RESULT")" "$CHECKPOINT_DIR"
        exec 9>"${RESULT}.lock"
        flock -n 9 || fail "evaluation already running: $RESULT"
        if [[ ! -f "$RESULT" ]]; then
            python -u "$ROOT_DIR/CV-DD/validate/train_hard_label_v1.py" \
                --train-manifest "$MANIFEST" --val-dir "$DATA_DIR/test" \
                --dataset-name "$DATASET" --num-classes "$CLASSES" --ipc 2 \
                --student-seed "$STUDENT_SEED" --result "$RESULT" \
                --checkpoint-dir "$CHECKPOINT_DIR" \
                --imagenet-weights-path "$TORCHVISION_MODEL_ROOT/resnet18-f37072fd.pth" \
                --total-updates 3000 --batch-size 64 --backbone-lr 3e-4 --head-lr 3e-3 \
                --backbone-min-lr 0 --head-min-lr 0 --momentum 0.9 \
                --weight-decay 5e-4 --eval-every-updates 300 \
                --workers 8 --persistent-workers --val-batch-size 256
        fi
        python "$ROOT_DIR/CV-DD/fine_grained/annotate_dino_ipc2_result.py" \
            --result "$RESULT" --selection-manifest "$MANIFEST" \
            --selection-audit "$SELECTION_AUDIT"
        python "$ROOT_DIR/CV-DD/fine_grained/audit_hard_label_v1_result.py" \
            --result "$RESULT" --dataset "$DATASET" --classes "$CLASSES" \
            --ipc 2 --student-seed "$STUDENT_SEED" --validation-images "$VAL_IMAGES"
        ;;
    *) fail "unknown stage: $STAGE" ;;
esac
