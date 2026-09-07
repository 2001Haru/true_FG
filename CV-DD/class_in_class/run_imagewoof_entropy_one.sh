#!/usr/bin/env bash
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export ENTROPY_DATASET_PROFILE=imagewoof
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM="${1:?usage: run_imagewoof_entropy_one.sh LAMBDA|top10 SELECTION_SEED STUDENT_SEED hard|soft1}"
SELECTION_SEED="${2:?missing selection seed}"
STUDENT_SEED="${3:?missing Student seed}"
SUPERVISION="${4:?missing hard|soft1}"
[[ "$ARM" == top10 || "$ARM" =~ ^(-4|-2|0|2|4|8|16|32)$ ]] || { echo "invalid arm" >&2; exit 2; }
[[ "$SELECTION_SEED" =~ ^(0|1|2)$ ]] || { echo "invalid selection seed" >&2; exit 2; }
[[ "$STUDENT_SEED" =~ ^(42|43|44)$ ]] || { echo "invalid Student seed" >&2; exit 2; }
[[ "$SUPERVISION" == hard || "$SUPERVISION" == soft1 ]] || { echo "invalid supervision" >&2; exit 2; }

DATA_ROOT="${IMAGEWOOF_DATA_ROOT:-/linxi/dataset/CV-DD/test_data/imagewoof}"
EXP_ROOT="${IMAGEWOOF_ENTROPY_ROOT:-/linxi/dataset/CV-DD/experiments/imagewoof_entropy_selection_v1}"
TEACHER="${IMAGEWOOF_TEACHER:?set IMAGEWOOF_TEACHER}"
SPEC="$ROOT/class_in_class/imagewoof_entropy_selection_protocol.json"
EVAL_EPOCHS=(200 400 600 800 1000 1200 1333 1400 1600 1666 1800 2000)

if [[ "$ARM" == top10 ]]; then
    MANIFEST="$EXP_ROOT/manifests/highest_entropy_top10_per_class.json"
    GATE="$EXP_ROOT/preflight/lambda0_entropy_allocation_gate.json"
    RESULT="$EXP_ROOT/results/highest_entropy_top10/${SUPERVISION}_sseed${STUDENT_SEED}.json"
    CHECKPOINT_DIR="$EXP_ROOT/checkpoints/highest_entropy_top10/${SUPERVISION}_sseed${STUDENT_SEED}"
else
    MANIFEST="$EXP_ROOT/manifests/lambda_$(printf '%+d' "$ARM")_rseed${SELECTION_SEED}.json"
    if (( ARM <= 4 )); then GATE="$EXP_ROOT/preflight/manual_visual_gate.json"; else GATE="$EXP_ROOT/preflight/high_lambda_extension_gate.json"; fi
    RESULT="$EXP_ROOT/results/lambda_$(printf '%+d' "$ARM")/rseed${SELECTION_SEED}/${SUPERVISION}_sseed${STUDENT_SEED}.json"
    CHECKPOINT_DIR="$EXP_ROOT/checkpoints/lambda_$(printf '%+d' "$ARM")/rseed${SELECTION_SEED}/${SUPERVISION}_sseed${STUDENT_SEED}"
fi
for path in "$DATA_ROOT/train" "$DATA_ROOT/test" "$TEACHER" "$SPEC" "$MANIFEST" "$GATE"; do
    [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 1; }
done
python - "$GATE" "$TEACHER" "$MANIFEST" "$ARM" "$SELECTION_SEED" <<'PY'
import hashlib,json,sys
from pathlib import Path
gate=json.load(open(sys.argv[1]));teacher=Path(sys.argv[2]);manifest=Path(sys.argv[3]);arm=sys.argv[4];seed=sys.argv[5]
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert gate.get('approved') is True and gate.get('teacher_checkpoint_sha256')==h(teacher)
if arm=='top10': assert gate.get('highest_entropy_top10_manifest_sha256')==h(manifest)
else: assert gate.get('manifest_sha256',{}).get(f'lambda_{int(arm):+d}_rseed{seed}')==h(manifest)
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
        --protocol-name imagewoof_entropy_selection_v1 --protocol-spec "$SPEC"
fi
if [[ "$ARM" == top10 ]]; then
    python "$ROOT/class_in_class/audit_imagenette_entropy_allocation_result.py" \
        --result "$RESULT" --arm "top10_${SUPERVISION}" --student-seed "$STUDENT_SEED" \
        --experiment-root "$EXP_ROOT" --protocol-name imagewoof_entropy_selection_v1 \
        --protocol-spec "$SPEC"
else
    python "$ROOT/class_in_class/audit_imagenette_entropy_result.py" \
        --result "$RESULT" --lambda "$ARM" --selection-seed "$SELECTION_SEED" \
        --student-seed "$STUDENT_SEED" --supervision "$SUPERVISION" \
        --protocol-name imagewoof_entropy_selection_v1 --protocol-spec "$SPEC" \
        --expected-eval-epochs "${EVAL_EPOCHS[@]}"
fi
