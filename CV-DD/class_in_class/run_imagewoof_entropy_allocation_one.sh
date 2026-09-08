#!/usr/bin/env bash
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export ENTROPY_DATASET_PROFILE=imagewoof
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM="${1:?usage: run_imagewoof_entropy_allocation_one.sh high_entropy_soft1|low_entropy_soft1 SELECTION_SEED STUDENT_SEED}"
SELECTION_SEED="${2:?missing selection seed}"
STUDENT_SEED="${3:?missing Student seed}"
[[ "$ARM" == high_entropy_soft1 || "$ARM" == low_entropy_soft1 ]] || { echo "invalid arm" >&2; exit 2; }
[[ "$SELECTION_SEED" =~ ^(0|1|2)$ ]] || { echo "invalid selection seed" >&2; exit 2; }
[[ "$STUDENT_SEED" =~ ^(42|43|44)$ ]] || { echo "invalid Student seed" >&2; exit 2; }

DATA_ROOT="${IMAGEWOOF_DATA_ROOT:-/linxi/dataset/CV-DD/test_data/imagewoof}"
MASTER_ROOT="${IMAGEWOOF_MASTER_ROOT:-/linxi/dataset/CV-DD/experiments/imagewoof_cvdd_v1}"
TEACHER="${IMAGEWOOF_TEACHER:-$MASTER_ROOT/teacher/workers8_eval_sparse_run2/ResNet18.pth}"
EXP_ROOT="${IMAGEWOOF_ENTROPY_ROOT:-/linxi/dataset/CV-DD/experiments/imagewoof_entropy_selection_v1}"
AUDIT="$EXP_ROOT/preflight/lambda0_entropy_allocation_audit.json"
GATE="$EXP_ROOT/preflight/lambda0_entropy_allocation_gate.json"
SPEC="$ROOT/class_in_class/imagewoof_entropy_allocation_protocol.json"
MANIFEST="$EXP_ROOT/manifests/lambda_+0_rseed${SELECTION_SEED}.json"
RESULT="$EXP_ROOT/results/lambda0_entropy_allocation/rseed${SELECTION_SEED}/${ARM}_sseed${STUDENT_SEED}.json"
CHECKPOINT_DIR="$EXP_ROOT/checkpoints/lambda0_entropy_allocation/rseed${SELECTION_SEED}/${ARM}_sseed${STUDENT_SEED}"
EVAL_EPOCHS=(200 400 600 800 1000 1200 1333 1400 1600 1666 1800 2000)
for path in "$DATA_ROOT/train" "$DATA_ROOT/test" "$TEACHER" "$AUDIT" "$GATE" "$SPEC" "$MANIFEST"; do
    [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 1; }
done
python - "$AUDIT" "$GATE" "$TEACHER" "$MANIFEST" "$SELECTION_SEED" <<'PY'
import hashlib,json,sys
from pathlib import Path
a=Path(sys.argv[1]);g=Path(sys.argv[2]);teacher=Path(sys.argv[3]);manifest=Path(sys.argv[4]);seed=sys.argv[5]
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit=json.load(open(a));gate=json.load(open(g))
assert audit['status']=='complete_pending_allocation_review' and gate.get('approved') is True
assert gate.get('audit_file')==str(a.resolve()) and gate.get('audit_sha256')==h(a)
assert gate.get('teacher_checkpoint_sha256')==h(teacher)
assert audit['lambda0_manifests'][seed]['sha256']==h(manifest)
PY
mkdir -p "$(dirname "$RESULT")" "$CHECKPOINT_DIR"
exec 9>"${RESULT}.lock"
flock -n 9 || { echo "result already active: $RESULT" >&2; exit 1; }
if [[ ! -f "$RESULT" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?set one GPU}" \
    python -u "$ROOT/class_in_class/train_imagenette_entropy_selection.py" \
        --train-manifest "$MANIFEST" --test-dir "$DATA_ROOT/test" \
        --teacher-checkpoint "$TEACHER" --supervision "$ARM" \
        --student-seed "$STUDENT_SEED" --result "$RESULT" --checkpoint-dir "$CHECKPOINT_DIR" \
        --workers "${ENTROPY_EVAL_WORKERS:-4}" --batch-size 64 --epochs 2000 \
        --eval-epochs "${EVAL_EPOCHS[@]}" --protocol-name imagewoof_entropy_allocation_v1 \
        --protocol-spec "$SPEC"
fi
python "$ROOT/class_in_class/audit_imagenette_entropy_allocation_result.py" \
    --result "$RESULT" --arm "$ARM" --selection-seed "$SELECTION_SEED" \
    --student-seed "$STUDENT_SEED" --experiment-root "$EXP_ROOT" \
    --protocol-name imagewoof_entropy_allocation_v1 --protocol-spec "$SPEC"
