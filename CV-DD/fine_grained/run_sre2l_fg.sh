#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:?usage: run_sre2l_fg.sh STAGE DATASET RUN_SEED [IPC] [STUDENT_SEED]}"
DATASET="${2:?missing dataset}"
RUN_SEED="${3:-42}"
IPC="${4:-}"
STUDENT_SEED="${5:-}"

DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FD2}"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1}"
TEACHER_SEED="${TEACHER_SEED:-42}"
PATCH_SEED="${PATCH_SEED:-42}"
WORKERS="${WORKERS:-4}"
RELABEL_WORKERS="${RELABEL_WORKERS:-2}"
EVAL_WORKERS="${EVAL_WORKERS:-2}"
STUDENT_TEMPERATURE="${STUDENT_TEMPERATURE:-20}"
RESULT_ROOT="${RESULT_ROOT:-$EXP_ROOT/results}"
POST_EVAL_ROOT="${POST_EVAL_ROOT:-$EXP_ROOT/post_eval}"

case "$DATASET" in
    CUB_imsize224|cub)
        DATASET=CUB_imsize224; CLASSES=200; VAL_IMAGES=5794; RECOVERY_ITERATIONS=10000; FKD_BATCH=20; ACCUMULATION=2 ;;
    A_imsize224|aircraft|a)
        DATASET=A_imsize224; CLASSES=100; VAL_IMAGES=3333; RECOVERY_ITERATIONS=4000; FKD_BATCH=20; ACCUMULATION=2 ;;
    SC_imsize224|cars|sc)
        DATASET=SC_imsize224; CLASSES=196; VAL_IMAGES=8041; RECOVERY_ITERATIONS=4000; FKD_BATCH=14; ACCUMULATION=2 ;;
    *) echo "Unsupported dataset: $DATASET" >&2; exit 2 ;;
esac

SOURCE_DATA_DIR="$DATA_ROOT/$DATASET"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-$EXP_ROOT/datasets}"
DATA_DIR="$PREPARED_DATA_ROOT/$DATASET"
TEACHER_DIR="$EXP_ROOT/teachers/$DATASET/tseed${TEACHER_SEED}"
TEACHER="$TEACHER_DIR/ResNet18.pth"
PATCH_BASE="$EXP_ROOT/patches/$DATASET/tseed${TEACHER_SEED}_pseed${PATCH_SEED}"
PATCH_DIR="$PATCH_BASE/2"
RECOVERY_ROOT="$EXP_ROOT/recovery/$DATASET/rseed${RUN_SEED}"
SYN_IPC5="$RECOVERY_ROOT/ipc5"
LOG_ROOT="$EXP_ROOT/logs/$DATASET"
mkdir -p "$EXP_ROOT" "$LOG_ROOT"

fail() { echo "PRECHECK FAILED: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || fail "missing file: $1"; }
require_dir() { [[ -d "$1" ]] || fail "missing directory: $1"; }

syn_for_ipc() {
    local ipc="$1"
    if [[ "$ipc" == 5 ]]; then printf '%s\n' "$SYN_IPC5"; else printf '%s\n' "$RECOVERY_ROOT/ipc${ipc}"; fi
}

fkd_base_for_ipc() { printf '%s\n' "$EXP_ROOT/fkd/$DATASET/rseed${RUN_SEED}/ipc${1}"; }
fkd_actual_for_ipc() { printf '%s\n' "$(fkd_base_for_ipc "$1")_bs${FKD_BATCH}_ipc${1}"; }

case "$STAGE" in
    prepare)
        python -u "$ROOT_DIR/fine_grained/prepare_resized_data.py" \
            --dataset-name "$DATASET" --source-dir "$SOURCE_DATA_DIR" \
            --output-dir "$DATA_DIR" --workers "${RESIZE_WORKERS:-16}" --skip-completed
        ;;

    audit)
        python -u "$ROOT_DIR/fine_grained/audit_inputs.py" \
            --dataset-name "$DATASET" --data-dir "$DATA_DIR" --verify-all-sizes \
            --output "$EXP_ROOT/audits/${DATASET}.json"
        ;;

    teacher)
        python -u "$ROOT_DIR/fine_grained/train_plain_teacher.py" \
            --dataset-name "$DATASET" --data-dir "$DATA_DIR" \
            --output-dir "$EXP_ROOT/teachers/$DATASET/tseed${RUN_SEED}" \
            --seed "$RUN_SEED" --epochs 51 --batch-size 4 --workers "$WORKERS" \
            --lr 1e-3 --momentum 0.9 --weight-decay 1e-5 \
            --deterministic --resume --skip-completed
        ;;

    patches)
        require_file "$TEACHER"
        python -u "$ROOT_DIR/fine_grained/make_patches.py" \
            --dataset-name "$DATASET" --data-dir "$DATA_DIR" --teacher "$TEACHER" \
            --output-dir "$PATCH_DIR" --seed "$RUN_SEED" \
            --patches-per-class 5 --num-crops 5 --skip-completed
        ;;

    recover)
        require_file "$TEACHER"
        require_dir "$PATCH_DIR"
        TEACHER_GATE="$TEACHER_DIR/teacher_gate.json"
        PATCH_MANIFEST="$PATCH_DIR/patch_manifest.json"
        require_file "$TEACHER_GATE"
        require_file "$PATCH_MANIFEST"
        patch_count="$(find "$PATCH_DIR" -type f -name '*.jpg' | wc -l)"
        [[ "$patch_count" -eq $((CLASSES * 5)) ]] || fail "patch count $patch_count != $((CLASSES * 5))"
        mkdir -p "$RECOVERY_ROOT"
        RECOVERY_MANIFEST="$RECOVERY_ROOT/recovery_manifest.json"
        python "$ROOT_DIR/fine_grained/record_recovery_manifest.py" \
            --dataset-name "$DATASET" --recovery-seed "$RUN_SEED" \
            --teacher "$TEACHER" --teacher-gate "$TEACHER_GATE" \
            --patch-dir "$PATCH_DIR" --patch-manifest "$PATCH_MANIFEST" \
            --output "$RECOVERY_MANIFEST" --status running
        python -u "$ROOT_DIR/recover/recover.py" \
            --exp-name ipc5 --apply-data-augmentation --dataset-name "$DATASET" \
            --batch-size 100 --syn-data-path "$RECOVERY_ROOT" --patch-dir "$PATCH_BASE" \
            --model-pool-dir "$TEACHER_DIR" --pretrained-model-type offline \
            --model-setting 0 --sre2l-model ResNet18 --voter-type equal --selected-size 1 \
            --lr 1e-3 --iteration "$RECOVERY_ITERATIONS" --r-bn 1e-3 \
            --first-bn-multiplier 10 --jitter 32 --seed "$RUN_SEED" \
            --store-best-images --skip-completed --ipc-start 0 --ipc-end 5 \
            --initialisation-method Patches --patch-diff 2
        count="$(find "$SYN_IPC5" -type f -name '*.jpg' | wc -l)"
        [[ "$count" -eq $((CLASSES * 5)) ]] || fail "IPC5 count $count != $((CLASSES * 5))"
        python "$ROOT_DIR/fine_grained/record_recovery_manifest.py" \
            --dataset-name "$DATASET" --recovery-seed "$RUN_SEED" \
            --teacher "$TEACHER" --teacher-gate "$TEACHER_GATE" \
            --patch-dir "$PATCH_DIR" --patch-manifest "$PATCH_MANIFEST" \
            --output "$RECOVERY_MANIFEST" --status complete
        ;;

    sample)
        require_dir "$SYN_IPC5"
        for target_ipc in 1 3; do
            target="$RECOVERY_ROOT/ipc${target_ipc}"
            count=0
            [[ -d "$target" ]] && count="$(find "$target" -type f -name '*.jpg' | wc -l)"
            if [[ "$count" -eq 0 ]]; then
                python "$ROOT_DIR/tools/sample_ipc.py" \
                    --source "$SYN_IPC5" --target "$target" \
                    --ipc "$target_ipc" --classes "$CLASSES"
            elif [[ "$count" -ne $((CLASSES * target_ipc)) ]]; then
                fail "$target contains $count images; expected $((CLASSES * target_ipc))"
            fi
        done
        python "$ROOT_DIR/fine_grained/audit_recovery_output.py" \
            --dataset-name "$DATASET" --recovery-root "$RECOVERY_ROOT"
        ;;

    relabel)
        [[ "$IPC" =~ ^(1|3|5)$ ]] || fail "relabel requires IPC 1, 3, or 5"
        require_file "$TEACHER"
        syn="$(syn_for_ipc "$IPC")"
        require_dir "$syn"
        fkd_base="$(fkd_base_for_ipc "$IPC")"
        fkd_actual="$(fkd_actual_for_ipc "$IPC")"
        expected_batches=$((400 * CLASSES * IPC / FKD_BATCH))
        batch_count=0
        [[ -d "$fkd_actual" ]] && batch_count="$(find "$fkd_actual" -type f -name 'batch_*.tar' | wc -l)"
        if [[ "$batch_count" -ne "$expected_batches" ]]; then
            mkdir -p "$(dirname "$fkd_base")"
            python -u "$ROOT_DIR/relabel/relabel.py" \
                --syn-data-path "$syn" --fkd-path "$fkd_base" \
                --model-pool-dir "$TEACHER_DIR" --teacher-model-name ResNet18 --gpu 0 \
                --batch-size "$FKD_BATCH" --workers "$RELABEL_WORKERS" \
                --dataset-name "$DATASET" --epochs 400 --seed 42 --fkd-seed 42 \
                --min-scale-crops 0.08 --max-scale-crops 1 --use-fp16 \
                --mode fkd_save --mix-type cutmix
        fi
        batch_count="$(find "$fkd_actual" -type f -name 'batch_*.tar' | wc -l)"
        [[ "$batch_count" -eq "$expected_batches" ]] || fail "FKD batches $batch_count != $expected_batches"
        python "$ROOT_DIR/fine_grained/audit_fkd.py" \
            --fkd-dir "$fkd_actual" --images $((CLASSES * IPC)) \
            --classes "$CLASSES" --batch-size "$FKD_BATCH" --epochs 400
        ;;

    eval)
        [[ "$IPC" =~ ^(1|3|5)$ ]] || fail "eval requires IPC 1, 3, or 5"
        [[ "$STUDENT_SEED" =~ ^[0-9]+$ ]] || fail "eval requires a numeric student seed"
        syn="$(syn_for_ipc "$IPC")"
        fkd_actual="$(fkd_actual_for_ipc "$IPC")"
        require_dir "$syn"
        require_dir "$fkd_actual"
        require_file "$fkd_actual/fkd_audit.json"
        result="$RESULT_ROOT/$DATASET/rseed${RUN_SEED}/ipc${IPC}_sseed${STUDENT_SEED}.json"
        mkdir -p "$(dirname "$result")"
        if [[ -f "$result" ]]; then
            python "$ROOT_DIR/fine_grained/audit_result.py" \
                --result "$result" --classes "$CLASSES" --validation-images "$VAL_IMAGES"
            echo "Evaluation already complete: $result"
            exit 0
        fi
        python -u "$ROOT_DIR/validate/train_fkd.py" \
            --model ResNet18 --ipc "$IPC" \
            --exp-name "sre2l_${DATASET}_ipc${IPC}_rseed${RUN_SEED}_sseed${STUDENT_SEED}" \
            --original-data-path "$syn" --fkd-path "$fkd_actual" \
            --output-dir "$POST_EVAL_ROOT" --batch-size "$FKD_BATCH" --epochs 400 \
            --dataset-name "$DATASET" --gradient-accumulation-steps "$ACCUMULATION" \
            --mix-type cutmix --cos --workers "$EVAL_WORKERS" --fkd_seed 42 \
            --train-seed "$STUDENT_SEED" --temperature "$STUDENT_TEMPERATURE" \
            --adamw-lr-override 1e-3 --adamw-weight-decay 1e-5 --eta-override 2 \
            --val-dir "$DATA_DIR/test" --disable-wandb --per-class-output "$result"
        python "$ROOT_DIR/fine_grained/audit_result.py" \
            --result "$result" --classes "$CLASSES" --validation-images "$VAL_IMAGES"
        ;;

    *)
        echo "Unknown stage $STAGE; use prepare, audit, teacher, patches, recover, sample, relabel, or eval" >&2
        exit 2
        ;;
esac
