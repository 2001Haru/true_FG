#!/usr/bin/env bash
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAMBDA="${1:?usage: run_imagenette_entropy_selection.sh LAMBDA SELECTION_SEED STUDENT_SEED hard|soft1}"
SELECTION_SEED="${2:?missing selection seed}"
STUDENT_SEED="${3:?missing student seed}"
SUPERVISION="${4:?missing supervision}"

[[ "$LAMBDA" =~ ^(-4|-2|0|2|4)$ ]] || { echo "invalid lambda: $LAMBDA" >&2; exit 2; }
[[ "$SELECTION_SEED" =~ ^(0|1|2)$ ]] || { echo "invalid selection seed" >&2; exit 2; }
[[ "$STUDENT_SEED" =~ ^(42|43|44)$ ]] || { echo "invalid Student seed" >&2; exit 2; }
[[ "$SUPERVISION" == hard || "$SUPERVISION" == soft1 ]] || { echo "first-round supervision must be hard or soft1" >&2; exit 2; }

DATA_ROOT="${IMAGENETTE_DATA_ROOT:-/linxi/dataset/CV-DD/test_data/imagenet-nette}"
TEACHER="${IMAGENETTE_C1_TEACHER:-/linxi/dataset/CV-DD/offline_models/imagenet-nette/ResNet18.pth}"
EXP_ROOT="${IMAGENETTE_ENTROPY_ROOT:-/linxi/dataset/CV-DD/experiments/imagenette_entropy_selection_v1}"
PREFLIGHT="$EXP_ROOT/preflight/selection_preflight.json"
VISUAL_GATE="$EXP_ROOT/preflight/manual_visual_gate.json"
ARM="lambda_$(printf '%+d' "$LAMBDA")_rseed${SELECTION_SEED}"
MANIFEST="$EXP_ROOT/manifests/$ARM.json"
RESULT="$EXP_ROOT/results/lambda_$(printf '%+d' "$LAMBDA")/rseed${SELECTION_SEED}/${SUPERVISION}_sseed${STUDENT_SEED}.json"
CHECKPOINT_DIR="$EXP_ROOT/checkpoints/lambda_$(printf '%+d' "$LAMBDA")/rseed${SELECTION_SEED}/${SUPERVISION}_sseed${STUDENT_SEED}"

for path in "$DATA_ROOT/train" "$DATA_ROOT/test" "$TEACHER" "$PREFLIGHT" "$VISUAL_GATE" "$MANIFEST"; do
    [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 1; }
done
python - "$PREFLIGHT" "$VISUAL_GATE" "$TEACHER" <<'PY'
import hashlib,json,sys
from pathlib import Path
preflight=Path(sys.argv[1]); gate=Path(sys.argv[2]); teacher=Path(sys.argv[3])
p=json.load(open(preflight)); g=json.load(open(gate))
h=hashlib.sha256(teacher.read_bytes()).hexdigest()
assert p['status']=='complete_pending_manual_visual_review'
assert g.get('approved') is True and g.get('preflight_file')==str(preflight.resolve())
assert g.get('teacher_checkpoint_sha256')==h==p['teacher_checkpoint_sha256']
assert isinstance(g.get('reviewer'),str) and g['reviewer']
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
        --batch-size 64 --epochs 2000 --eval-every-epochs 20
fi
python "$ROOT/class_in_class/audit_imagenette_entropy_result.py" \
    --result "$RESULT" --lambda "$LAMBDA" --selection-seed "$SELECTION_SEED" \
    --student-seed "$STUDENT_SEED" --supervision "$SUPERVISION"

