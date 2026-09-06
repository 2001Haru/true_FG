#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${1:?usage: summarize_imagenette_entropy_phase.sh endpoints|full}"
EXP_ROOT="${IMAGENETTE_ENTROPY_ROOT:-/linxi/dataset/CV-DD/experiments/imagenette_entropy_selection_v1}"
if [[ "$PHASE" == endpoints ]]; then
    LAMBDAS=(-4 0 4); REQUIRED_PHASE=endpoints; OUTPUT="$EXP_ROOT/summary/endpoints.json"
elif [[ "$PHASE" == full ]]; then
    LAMBDAS=(-4 -2 0 2 4); REQUIRED_PHASE=middle; OUTPUT="$EXP_ROOT/summary/full.json"
else
    echo "phase must be endpoints or full" >&2; exit 2
fi
for node in 0 1; do
    status="$EXP_ROOT/status/${REQUIRED_PHASE}_node${node}.json"
    [[ -f "$status" ]] && [[ "$(python -c "import json;print(json.load(open('$status'))['status'])")" == complete ]] \
        || { echo "incomplete shard: $status" >&2; exit 1; }
done
mkdir -p "$(dirname "$OUTPUT")"
python "$ROOT/class_in_class/summarize_imagenette_entropy_selection.py" \
    --experiment-root "$EXP_ROOT" --lambdas "${LAMBDAS[@]}" --output "$OUTPUT"

