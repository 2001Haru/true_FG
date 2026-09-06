#!/usr/bin/env bash
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM="${1:?usage: run_imagenette_entropy_allocation.sh ARM SELECTION_SEED STUDENT_SEED}"
SELECTION_SEED="${2:?missing selection seed}"
STUDENT_SEED="${3:?missing student seed}"
[[ "$ARM" =~ ^(high_entropy_soft1|low_entropy_soft1|top10_hard|top10_soft1)$ ]] || { echo "invalid arm" >&2; exit 2; }
[[ "$SELECTION_SEED" =~ ^(0|1|2)$ ]] || { echo "invalid selection seed" >&2; exit 2; }
[[ "$STUDENT_SEED" =~ ^(42|43|44)$ ]] || { echo "invalid Student seed" >&2; exit 2; }

DATA_ROOT="${IMAGENETTE_DATA_ROOT:-/linxi/dataset/CV-DD/test_data/imagenet-nette}"
TEACHER="${IMAGENETTE_C1_TEACHER:-/linxi/dataset/CV-DD/offline_models/imagenet-nette/ResNet18.pth}"
EXP_ROOT="${IMAGENETTE_ENTROPY_ROOT:-/linxi/dataset/CV-DD/experiments/imagenette_entropy_selection_v1}"
AUDIT="$EXP_ROOT/preflight/lambda0_entropy_allocation_audit.json"
GATE="$EXP_ROOT/preflight/lambda0_entropy_allocation_gate.json"
SPEC="$ROOT/class_in_class/imagenette_entropy_allocation_protocol.json"
EVAL_EPOCHS=(200 400 600 800 1000 1200 1333 1400 1600 1666 1800 2000)

if [[ "$ARM" == top10_* ]]; then
    MANIFEST="$EXP_ROOT/manifests/highest_entropy_top10_per_class.json"
    SUPERVISION="${ARM#top10_}"
    RESULT="$EXP_ROOT/results/highest_entropy_top10/${SUPERVISION}_sseed${STUDENT_SEED}.json"
    CHECKPOINT_DIR="$EXP_ROOT/checkpoints/highest_entropy_top10/${SUPERVISION}_sseed${STUDENT_SEED}"
else
    MANIFEST="$EXP_ROOT/manifests/lambda_$(printf '%+d' 0)_rseed${SELECTION_SEED}.json"
    SUPERVISION="$ARM"
    RESULT="$EXP_ROOT/results/lambda0_entropy_allocation/rseed${SELECTION_SEED}/${ARM}_sseed${STUDENT_SEED}.json"
    CHECKPOINT_DIR="$EXP_ROOT/checkpoints/lambda0_entropy_allocation/rseed${SELECTION_SEED}/${ARM}_sseed${STUDENT_SEED}"
fi

for path in "$DATA_ROOT/train" "$DATA_ROOT/test" "$TEACHER" "$AUDIT" "$GATE" "$SPEC" "$MANIFEST"; do
    [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 1; }
done
python - "$AUDIT" "$GATE" <<'PY'
import hashlib,json,sys
from pathlib import Path
a=Path(sys.argv[1]).resolve(); g=Path(sys.argv[2]).resolve()
audit=json.load(open(a)); gate=json.load(open(g))
assert audit['status']=='complete_pending_allocation_review'
assert gate.get('approved') is True and gate.get('audit_file')==str(a)
assert gate.get('audit_sha256')==hashlib.sha256(a.read_bytes()).hexdigest()
assert isinstance(gate.get('reviewer'),str) and gate['reviewer']
PY

mkdir -p "$(dirname "$RESULT")" "$CHECKPOINT_DIR"
exec 9>"${RESULT}.lock"
flock -n 9 || { echo "result already running: $RESULT" >&2; exit 1; }
if [[ ! -f "$RESULT" ]]; then
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?set one GPU}" \
    python -u "$ROOT/class_in_class/train_imagenette_entropy_selection.py" \
        --train-manifest "$MANIFEST" --test-dir "$DATA_ROOT/test" \
        --teacher-checkpoint "$TEACHER" --supervision "$SUPERVISION" \
        --student-seed "$STUDENT_SEED" --result "$RESULT" \
        --checkpoint-dir "$CHECKPOINT_DIR" --workers "${ENTROPY_EVAL_WORKERS:-4}" \
        --batch-size 64 --epochs 2000 --eval-epochs "${EVAL_EPOCHS[@]}" \
        --protocol-name imagenette_entropy_allocation_v1 --protocol-spec "$SPEC"
fi
AUDIT_ARGS=(--result "$RESULT" --arm "$ARM" --student-seed "$STUDENT_SEED" --experiment-root "$EXP_ROOT")
if [[ "$ARM" != top10_* ]]; then AUDIT_ARGS+=(--selection-seed "$SELECTION_SEED"); fi
python "$ROOT/class_in_class/audit_imagenette_entropy_allocation_result.py" "${AUDIT_ARGS[@]}"

