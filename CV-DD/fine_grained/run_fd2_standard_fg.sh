#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:?usage: run_fd2_standard_fg.sh STAGE SEMANTICS DATASET RUN_SEED [IPC] [STUDENT_SEED]}"
SEMANTICS="${2:?missing semantics: released_semantics or paper_literal}"
DATASET="${3:?missing dataset}"
RUN_SEED="${4:-42}"
IPC="${5:-}"
STUDENT_SEED="${6:-}"

case "$SEMANTICS" in
    released_semantics|paper_literal) ;;
    *) echo "Unsupported semantics: $SEMANTICS" >&2; exit 2 ;;
esac
case "$DATASET" in
    CUB_imsize224|cub)
        DATASET=CUB_imsize224; CLASSES=200; FKD_BATCH=20 ;;
    A_imsize224|aircraft|a)
        DATASET=A_imsize224; CLASSES=100; FKD_BATCH=20 ;;
    SC_imsize224|cars|sc)
        DATASET=SC_imsize224; CLASSES=196; FKD_BATCH=14 ;;
    *) echo "Unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FD2}"
FD2_ROOT="${FD2_ROOT:-/linxi/dataset/FG_FD2_standard/v1}"
EXP_ROOT="$FD2_ROOT/$SEMANTICS"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-$FD2_ROOT/datasets}"
DATA_DIR="$PREPARED_DATA_ROOT/$DATASET"
TEACHER_SEED="${TEACHER_SEED:-$RUN_SEED}"
PATCH_SEED="${PATCH_SEED:-42}"
TEACHER_DIR="$EXP_ROOT/teachers/$DATASET/tseed${TEACHER_SEED}"
BACKBONE="$TEACHER_DIR/ResNet18.pth"
JOINT_TEACHER="$TEACHER_DIR/FD2_ResNet18_CAL.pth"
PATCH_BASE="$EXP_ROOT/patches/$DATASET/tseed${TEACHER_SEED}_pseed${PATCH_SEED}"
PATCH_DIR="$PATCH_BASE/2"
RECOVERY_ROOT="$EXP_ROOT/recovery/$DATASET/rseed${RUN_SEED}"
SYN_IPC5="$RECOVERY_ROOT/ipc5"
RESULT_ROOT="${RESULT_ROOT:-$EXP_ROOT/results}"
LOG_ROOT="$EXP_ROOT/logs/$DATASET"
mkdir -p "$LOG_ROOT"
LOG_FILE="$LOG_ROOT/${STAGE}_tseed${TEACHER_SEED}_rseed${RUN_SEED}${IPC:+_ipc${IPC}}${STUDENT_SEED:+_sseed${STUDENT_SEED}}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "stage=$STAGE semantics=$SEMANTICS dataset=$DATASET teacher_seed=$TEACHER_SEED recovery_seed=$RUN_SEED"
echo "log=$LOG_FILE"

fail() { echo "PRECHECK FAILED: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || fail "missing file: $1"; }
require_dir() { [[ -d "$1" ]] || fail "missing directory: $1"; }

delegate_plain_stage() {
    local delegated_stage="$1"
    shift
    DATA_ROOT="$DATA_ROOT" \
    EXP_ROOT="$EXP_ROOT" \
    PREPARED_DATA_ROOT="$PREPARED_DATA_ROOT" \
    TEACHER_DIR_OVERRIDE="$TEACHER_DIR" \
    TEACHER_SEED="$TEACHER_SEED" \
    PATCH_SEED="$PATCH_SEED" \
    RESULT_ROOT="$RESULT_ROOT" \
    RECOVERY_ITERATIONS_OVERRIDE=4000 \
    TEACHER_WORKERS=8 RELABEL_WORKERS=8 EVAL_WORKERS=8 EVAL_PERSISTENT_WORKERS=1 \
        bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" "$delegated_stage" "$DATASET" "$RUN_SEED" "$@"
}

case "$STAGE" in
    prepare|audit)
        delegate_plain_stage "$STAGE"
        ;;

    teacher)
        python -u "$ROOT_DIR/fine_grained/train_fd2_standard_teacher.py" \
            --semantics "$SEMANTICS" --dataset-name "$DATASET" \
            --data-dir "$DATA_DIR" --output-dir "$TEACHER_DIR" \
            --seed "$TEACHER_SEED" --epochs 100 --batch-size 32 --workers 8 \
            --lr 1e-2 --momentum 0.9 --weight-decay 1e-4 --eta-min 1e-5 \
            --center-weight 1 --skip-completed
        ;;

    patches)
        require_file "$BACKBONE"
        python -u "$ROOT_DIR/fine_grained/make_patches.py" \
            --dataset-name "$DATASET" --data-dir "$DATA_DIR" --teacher "$BACKBONE" \
            --output-dir "$PATCH_DIR" --seed "$PATCH_SEED" \
            --patches-per-class 5 --num-crops 5 --skip-completed
        ;;

    recover)
        require_file "$JOINT_TEACHER"
        require_dir "$PATCH_DIR"
        patch_count="$(find "$PATCH_DIR" -type f -name '*.jpg' | wc -l)"
        [[ "$patch_count" -eq $((CLASSES * 5)) ]] || fail "patch count $patch_count != $((CLASSES * 5))"
        python -u "$ROOT_DIR/recover/recover_fd2_standard.py" \
            --semantics "$SEMANTICS" --dataset-name "$DATASET" \
            --teacher "$JOINT_TEACHER" --output-dir "$SYN_IPC5" \
            --patch-dir "$PATCH_BASE" --seed "$RUN_SEED" \
            --iterations 4000 --batch-size 100 --lr 1e-3 --r-bn 1e-3 \
            --first-bn-multiplier 10 --jitter 32 --group-size 4 \
            --intra-feature-weight 0.5 --ipc-start 0 --ipc-end 5 \
            --apply-data-augmentation --skip-completed
        ;;

    sample|relabel|eval|eval-hard)
        require_file "$BACKBONE"
        if [[ "$STAGE" == sample ]]; then
            delegate_plain_stage sample
        elif [[ "$STAGE" == relabel ]]; then
            [[ "$IPC" =~ ^(1|3|5)$ ]] || fail "relabel requires IPC 1, 3, or 5"
            delegate_plain_stage relabel "$IPC"
        else
            [[ "$IPC" =~ ^(1|3|5)$ ]] || fail "$STAGE requires IPC 1, 3, or 5"
            [[ "$STUDENT_SEED" =~ ^[0-9]+$ ]] || fail "$STAGE requires a numeric student seed"
            delegate_plain_stage "$STAGE" "$IPC" "$STUDENT_SEED"
            result="$RESULT_ROOT/$DATASET/rseed${RUN_SEED}/ipc${IPC}_sseed${STUDENT_SEED}.json"
            python "$ROOT_DIR/fine_grained/annotate_fd2_result.py" \
                --result "$result" --semantics "$SEMANTICS" \
                --joint-teacher "$JOINT_TEACHER" --backbone "$BACKBONE" \
                --recovery-manifest "$RECOVERY_ROOT/recovery_manifest.json"
        fi
        ;;

    *)
        echo "Unknown stage $STAGE; use prepare, audit, teacher, patches, recover, sample, relabel, eval, or eval-hard" >&2
        exit 2
        ;;
esac
