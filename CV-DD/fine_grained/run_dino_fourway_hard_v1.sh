#!/usr/bin/env bash
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="${1:?usage: run_dino_fourway_hard_v1.sh prepare|eval DATASET [ARM] [STUDENT_SEED]}"
DATASET="${2:?missing dataset}"
ARM="${3:-}"
STUDENT_SEED="${4:-}"

case "$DATASET" in
    CUB_imsize224|cub) DATASET=CUB_imsize224; SPEC=cub; CLASSES=200; VAL_IMAGES=5794 ;;
    A_imsize224|aircraft|a) DATASET=A_imsize224; SPEC=aircraft; CLASSES=100; VAL_IMAGES=3333 ;;
    SC_imsize224|cars|sc) DATASET=SC_imsize224; SPEC=cars; CLASSES=196; VAL_IMAGES=8041 ;;
    *) echo "Unknown dataset: $DATASET" >&2; exit 2 ;;
esac

DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
CODA_ROOT="${CODA_BASE_ROOT:-/linxi/dataset/FG_CoDA_standard/v2}"
DINO_MODEL_ROOT="${DINO_MODEL_ROOT:-/linxi/models/DINOv2/dinov2-base}"
EXP_ROOT="${DINO_FIVEARM_EXP_ROOT:-/linxi/dataset/FG_HardLabel_standard/v1/dino_fivearm_ipc1}"
DATA_DIR="$DATA_ROOT/$DATASET"
CACHE_DIR="$CODA_ROOT/shared_feature_cache/dinov2/$SPEC"
FEATURE_AUDIT="$CODA_ROOT/cache_audits/dinov2/$DATASET.json"
GENERATION_CONFIG="$CODA_ROOT/dino_space/generation/$DATASET/n5_mcs2_ms1/ipc5/gseed0/generation_config.json"
SELECTION_BASE="$EXP_ROOT/selections/$DATASET"
SELECTION_AUDIT="$EXP_ROOT/selection_audits/$DATASET.json"

fail() { echo "ERROR: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || fail "missing file: $1"; }

case "$STAGE" in
    prepare)
        python "$ROOT_DIR/CV-DD/fine_grained/prepare_dino_fourway_ipc1.py" \
            --data-dir "$DATA_DIR" --cache-dir "$CACHE_DIR" \
            --feature-audit "$FEATURE_AUDIT" --generation-config "$GENERATION_CONFIG" \
            --dino-model-root "$DINO_MODEL_ROOT" --repo-root "$ROOT_DIR" \
            --output-root "$SELECTION_BASE" --audit-output "$SELECTION_AUDIT" \
            --dataset-name "$DATASET" --classes "$CLASSES"
        ;;
    eval)
        case "$ARM" in
            centroid|rival_facing_edge|outward_edge|edge_high_margin)
                [[ "$STUDENT_SEED" =~ ^(42|43|44|45|46|47)$ ]] \
                    || fail "deterministic selection requires Student seed 42..47"
                ;;
            random_rseed0|random_rseed1|random_rseed2)
                [[ "$STUDENT_SEED" =~ ^(42|43|44)$ ]] \
                    || fail "random selection requires Student seed 42..44"
                ;;
            *) fail "unknown selection arm: $ARM" ;;
        esac
        MANIFEST="$SELECTION_BASE/manifests/$ARM.json"
        SELECTED_DIR="$SELECTION_BASE/$ARM"
        RESULT="$EXP_ROOT/results/$DATASET/$ARM/sseed${STUDENT_SEED}.json"
        CHECKPOINT_DIR="$EXP_ROOT/checkpoints/$DATASET/$ARM/sseed${STUDENT_SEED}"
        require_file "$MANIFEST"
        mkdir -p "$(dirname "$RESULT")" "$CHECKPOINT_DIR"
        exec 9>"${RESULT}.lock"
        flock -n 9 || fail "evaluation already running: $RESULT"
        if [[ ! -f "$RESULT" ]]; then
            python -u "$ROOT_DIR/CV-DD/validate/train_hard_label_v1.py" \
                --train-dir "$SELECTED_DIR" --val-dir "$DATA_DIR/test" \
                --dataset-name "$DATASET" --num-classes "$CLASSES" --ipc 1 \
                --student-seed "$STUDENT_SEED" --result "$RESULT" \
                --checkpoint-dir "$CHECKPOINT_DIR" --total-updates 3000 \
                --batch-size 64 --backbone-lr 3e-4 --head-lr 3e-3 \
                --backbone-min-lr 0 --head-min-lr 0 --momentum 0.9 \
                --weight-decay 5e-4 --eval-every-updates 300 \
                --workers 8 --persistent-workers --val-batch-size 256
        fi
        python "$ROOT_DIR/CV-DD/fine_grained/annotate_dino_selection_result.py" \
            --result "$RESULT" --selection-manifest "$MANIFEST"
        python "$ROOT_DIR/CV-DD/fine_grained/audit_hard_label_v1_result.py" \
            --result "$RESULT" --dataset "$DATASET" --classes "$CLASSES" \
            --ipc 1 --student-seed "$STUDENT_SEED" --validation-images "$VAL_IMAGES"
        ;;
    *) fail "unknown stage: $STAGE" ;;
esac
