#!/usr/bin/env bash
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GROUP="${1:?usage: run_imagenette_nontarget_permutation.sh high|low SELECTION_SEED STUDENT_SEED}"
RSEED="${2:?missing selection seed}"
SSEED="${3:?missing Student seed}"
[[ "$GROUP" == high || "$GROUP" == low ]] || { echo "group must be high or low" >&2; exit 2; }
[[ "$RSEED" =~ ^(0|1|2)$ ]] || { echo "invalid selection seed" >&2; exit 2; }
[[ "$SSEED" =~ ^(42|43|44)$ ]] || { echo "invalid Student seed" >&2; exit 2; }

DATA_ROOT="${IMAGENETTE_DATA_ROOT:-/linxi/dataset/CV-DD/test_data/imagenet-nette}"
TEACHER="${IMAGENETTE_C1_TEACHER:-/linxi/dataset/CV-DD/offline_models/imagenet-nette/ResNet18.pth}"
EXP_ROOT="${IMAGENETTE_ENTROPY_ROOT:-/linxi/dataset/CV-DD/experiments/imagenette_entropy_selection_v1}"
MANIFEST="$EXP_ROOT/manifests/lambda_$(printf '%+d' 0)_rseed${RSEED}.json"
PERMUTATIONS="$EXP_ROOT/preflight/lambda0_nontarget_permutations.json"
GATE="$EXP_ROOT/preflight/lambda0_nontarget_permutation_gate.json"
SPEC="$ROOT/class_in_class/imagenette_nontarget_permutation_protocol.json"
SUPERVISION="${GROUP}_entropy_permuted_soft1"
RESULT="$EXP_ROOT/results/lambda0_nontarget_permutation/rseed${RSEED}/${SUPERVISION}_sseed${SSEED}.json"
CHECKPOINT_DIR="$EXP_ROOT/checkpoints/lambda0_nontarget_permutation/rseed${RSEED}/${SUPERVISION}_sseed${SSEED}"
EVAL_EPOCHS=(200 400 600 800 1000 1200 1333 1400 1600 1666 1800 2000)

for path in "$DATA_ROOT/train" "$DATA_ROOT/test" "$TEACHER" "$MANIFEST" "$PERMUTATIONS" "$GATE" "$SPEC"; do
  [[ -e "$path" ]] || { echo "missing required input: $path" >&2; exit 1; }
done
python - "$PERMUTATIONS" "$GATE" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]).resolve(); g=Path(sys.argv[2]).resolve(); gate=json.load(open(g))
assert json.load(open(p))['status']=='complete'
assert gate.get('approved') is True and gate.get('permutation_file')==str(p)
assert gate.get('permutation_sha256')==hashlib.sha256(p.read_bytes()).hexdigest()
PY

mkdir -p "$(dirname "$RESULT")" "$CHECKPOINT_DIR"
exec 9>"${RESULT}.lock"
flock -n 9 || { echo "result already running: $RESULT" >&2; exit 1; }
if [[ ! -f "$RESULT" ]]; then
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?set one GPU}" \
  python -u "$ROOT/class_in_class/train_imagenette_entropy_selection.py" \
    --train-manifest "$MANIFEST" --test-dir "$DATA_ROOT/test" \
    --teacher-checkpoint "$TEACHER" --supervision "$SUPERVISION" \
    --non-target-permutation-file "$PERMUTATIONS" \
    --student-seed "$SSEED" --result "$RESULT" --checkpoint-dir "$CHECKPOINT_DIR" \
    --workers "${ENTROPY_EVAL_WORKERS:-4}" --batch-size 64 --epochs 2000 \
    --eval-epochs "${EVAL_EPOCHS[@]}" \
    --protocol-name imagenette_nontarget_permutation_v1 --protocol-spec "$SPEC"
fi
python "$ROOT/class_in_class/audit_imagenette_nontarget_permutation_result.py" \
  --result "$RESULT" --selection-seed "$RSEED" --student-seed "$SSEED" \
  --group "$GROUP" --experiment-root "$EXP_ROOT"

