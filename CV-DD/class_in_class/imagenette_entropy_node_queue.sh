#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${1:?usage: imagenette_entropy_node_queue.sh endpoints|middle NODE_INDEX}"
NODE_INDEX="${2:?missing node index 0 or 1}"
[[ "$PHASE" == endpoints || "$PHASE" == middle ]] || { echo "invalid phase" >&2; exit 2; }
[[ "$NODE_INDEX" == 0 || "$NODE_INDEX" == 1 ]] || { echo "node index must be 0 or 1" >&2; exit 2; }

EXP_ROOT="${IMAGENETTE_ENTROPY_ROOT:-/linxi/dataset/CV-DD/experiments/imagenette_entropy_selection_v1}"
LOG_ROOT="$EXP_ROOT/logs/$PHASE/node${NODE_INDEX}"
STATUS="$EXP_ROOT/status/${PHASE}_node${NODE_INDEX}.json"
GPU_COUNT="${ENTROPY_GPU_COUNT:-2}"
EVALS_PER_GPU="${ENTROPY_EVALS_PER_GPU:-2}"
WAVE_SIZE=$((GPU_COUNT * EVALS_PER_GPU))
GLOBAL_LOCK_ROOT="${GLOBAL_GPU_LOCK_ROOT:-$EXP_ROOT/locks}"
mkdir -p "$LOG_ROOT" "$(dirname "$STATUS")" "$GLOBAL_LOCK_ROOT"

if [[ "$PHASE" == endpoints ]]; then
    LAMBDAS=(-4 0 4)
else
    LAMBDAS=(-2 2)
    for node in 0 1; do
        p="$EXP_ROOT/status/endpoints_node${node}.json"
        [[ -f "$p" ]] && [[ "$(python -c "import json;print(json.load(open('$p'))['status'])")" == complete ]] \
            || { echo "middle phase requires both endpoint shards complete" >&2; exit 1; }
    done
fi

write_status() {
    python - "$STATUS" "$1" "${2:-0}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); x={'status':sys.argv[2],'exit_code':int(sys.argv[3]),'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}; t=p.with_suffix('.tmp'); t.write_text(json.dumps(x,indent=2)+'\n'); os.replace(t,p)
PY
}
on_exit(){ code=$?; (( code == 0 )) || write_status failed "$code"; }
trap on_exit EXIT
exec 9>"$GLOBAL_LOCK_ROOT/${PHASE}_node${NODE_INDEX}.lock"
flock -n 9 || { echo "queue already active for this node shard" >&2; exit 1; }
write_status running 0

wait_wave(){ local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; (( failed == 0 )); }
task_index=0; local_index=0; pids=()
for lambda in "${LAMBDAS[@]}"; do
  for supervision in hard soft1; do
    for rseed in 0 1 2; do
      for sseed in 42 43 44; do
        if (( task_index % 2 == NODE_INDEX )); then
          gpu=$((local_index % GPU_COUNT)); log="$LOG_ROOT/lambda_$(printf '%+d' "$lambda")_${supervision}_r${rseed}_s${sseed}.log"
          CUDA_VISIBLE_DEVICES="$gpu" bash "$ROOT/class_in_class/run_imagenette_entropy_selection.sh" \
              "$lambda" "$rseed" "$sseed" "$supervision" >"$log" 2>&1 &
          pids+=("$!"); local_index=$((local_index+1))
          if (( ${#pids[@]} == WAVE_SIZE )); then wait_wave "${pids[@]}"; pids=(); fi
        fi
        task_index=$((task_index+1))
      done
    done
  done
done
if (( ${#pids[@]} > 0 )); then wait_wave "${pids[@]}"; fi
write_status complete 0
trap - EXIT

