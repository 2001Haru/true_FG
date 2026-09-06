#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE_INDEX="${1:?usage: imagenette_entropy_auto_continue.sh NODE_INDEX}"
[[ "$NODE_INDEX" == 0 || "$NODE_INDEX" == 1 ]] || { echo "node index must be 0 or 1" >&2; exit 2; }
EXP_ROOT="${IMAGENETTE_ENTROPY_ROOT:-/linxi/dataset/CV-DD/experiments/imagenette_entropy_selection_v1}"
POLL_SECONDS="${ENTROPY_CONTINUE_POLL_SECONDS:-60}"

status_value() {
    local path="$1"
    [[ -f "$path" ]] || { echo missing; return; }
    python -c "import json;print(json.load(open('$path')).get('status','invalid'))"
}
wait_for_two_shards() {
    local phase="$1" first second
    while true; do
        first="$(status_value "$EXP_ROOT/status/${phase}_node0.json")"
        second="$(status_value "$EXP_ROOT/status/${phase}_node1.json")"
        if [[ "$first" == failed || "$second" == failed || "$first" == invalid || "$second" == invalid ]]; then
            echo "$phase failed: node0=$first node1=$second" >&2
            exit 1
        fi
        [[ "$first" == complete && "$second" == complete ]] && return
        sleep "$POLL_SECONDS"
    done
}

wait_for_two_shards endpoints
if [[ "$NODE_INDEX" == 0 ]]; then
    bash "$ROOT/class_in_class/summarize_imagenette_entropy_phase.sh" endpoints \
        >"$EXP_ROOT/logs/endpoints_summary.log" 2>&1
else
    while [[ ! -f "$EXP_ROOT/summary/endpoints.json" ]]; do sleep "$POLL_SECONDS"; done
fi

# The middle lambda values are pre-registered. No metric is read before launch.
bash "$ROOT/class_in_class/imagenette_entropy_node_queue.sh" middle "$NODE_INDEX"

if [[ "$NODE_INDEX" == 0 ]]; then
    wait_for_two_shards middle
    bash "$ROOT/class_in_class/summarize_imagenette_entropy_phase.sh" full \
        >"$EXP_ROOT/logs/full_summary.log" 2>&1
fi
