#!/usr/bin/env bash
set -euo pipefail
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM="${1:?usage: run_imagenette_ordering_hierarchy.sh O|B|A|Aprime MANIFEST_SEED STUDENT_SEED}"
RSEED="${2:?missing manifest seed}"; SSEED="${3:?missing student seed}"
[[ "$ARM" =~ ^(O|B|A|Aprime)$ ]] || { echo invalid arm >&2; exit 2; }
[[ "$RSEED" =~ ^(3|4|5|6|7|8|9|10|11)$ ]] || { echo invalid manifest >&2; exit 2; }
[[ "$SSEED" =~ ^(42|43|44)$ ]] || { echo invalid student >&2; exit 2; }
DATA_ROOT="${IMAGENETTE_DATA_ROOT:-/linxi/dataset/CV-DD/test_data/imagenet-nette}"
TEACHER="${IMAGENETTE_C1_TEACHER:-/linxi/dataset/CV-DD/offline_models/imagenet-nette/ResNet18.pth}"
EXP_ROOT="${IMAGENETTE_ORDERING_ROOT:-/linxi/dataset/CV-DD/experiments/imagenette_ordering_hierarchy_v1}"
MANIFEST="$EXP_ROOT/manifests/lambda_+0_rseed${RSEED}.json"
TEMPLATES="$EXP_ROOT/preflight/ordering_templates.json"
PREFLIGHT="$EXP_ROOT/preflight/ordering_preflight.json"
SPEC="$ROOT/class_in_class/imagenette_ordering_hierarchy_protocol.json"
RESULT="$EXP_ROOT/results/rseed${RSEED}/sseed${SSEED}/${ARM}.json"
CHECKPOINT="$EXP_ROOT/checkpoints/rseed${RSEED}/sseed${SSEED}/${ARM}"
for path in "$DATA_ROOT/test" "$TEACHER" "$MANIFEST" "$TEMPLATES" "$PREFLIGHT" "$SPEC"; do [[ -e "$path" ]] || { echo "missing $path" >&2; exit 1; }; done
python - "$PREFLIGHT" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]));assert p['status']=='complete_passed_training_gate' and p['training_gate']['passed'] is True
PY
mkdir -p "$(dirname "$RESULT")" "$CHECKPOINT"
exec 9>"${RESULT}.lock"; flock -n 9 || { echo already-running >&2; exit 1; }
if [[ ! -f "$RESULT" ]]; then
 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?set GPU}" python -u "$ROOT/class_in_class/train_imagenette_ordering_hierarchy.py" \
  --manifest "$MANIFEST" --templates "$TEMPLATES" --test-dir "$DATA_ROOT/test" --teacher-checkpoint "$TEACHER" \
  --protocol-spec "$SPEC" --arm "$ARM" --student-seed "$SSEED" --result "$RESULT" --checkpoint-dir "$CHECKPOINT" \
  --workers "${ORDERING_WORKERS:-4}"
fi
python "$ROOT/class_in_class/audit_imagenette_ordering_result.py" --result "$RESULT" --arm "$ARM" --selection-seed "$RSEED" --student-seed "$SSEED"

