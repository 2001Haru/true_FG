#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE_INDEX="${1:?usage: imagenette_entropy_high_lambda_queue.sh NODE_INDEX}"
[[ "$NODE_INDEX" == 0 || "$NODE_INDEX" == 1 ]] || { echo "node index must be 0 or 1" >&2; exit 2; }
EXP_ROOT="${IMAGENETTE_ENTROPY_ROOT:-/linxi/dataset/CV-DD/experiments/imagenette_entropy_selection_v1}"
LOG_ROOT="$EXP_ROOT/logs/high_lambda/node${NODE_INDEX}"
STATUS="$EXP_ROOT/status/high_lambda_node${NODE_INDEX}.json"
GPU_COUNT="${ENTROPY_GPU_COUNT:-2}"
EVALS_PER_GPU="${ENTROPY_EVALS_PER_GPU:-2}"
WAVE_SIZE=$((GPU_COUNT * EVALS_PER_GPU))
LOCK_ROOT="${GLOBAL_GPU_LOCK_ROOT:-$EXP_ROOT/locks}"
mkdir -p "$LOG_ROOT" "$(dirname "$STATUS")" "$LOCK_ROOT"

write_status() {
    python - "$STATUS" "$1" "${2:-0}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); x={'status':sys.argv[2],'exit_code':int(sys.argv[3]),'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}; t=p.with_suffix('.tmp'); t.write_text(json.dumps(x,indent=2)+'\n'); os.replace(t,p)
PY
}
on_exit(){ code=$?; (( code == 0 )) || write_status failed "$code"; }
exec 9>"$LOCK_ROOT/high_lambda_node${NODE_INDEX}.lock"
flock -n 9 || { echo "high-lambda queue already active for this node shard" >&2; exit 1; }
trap on_exit EXIT
write_status running 0

wait_wave(){ local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; (( failed == 0 )); }
task_index=0; local_index=0; pids=()
for lambda in 8 16 32; do
  for supervision in hard soft1; do
    for rseed in 0 1 2; do
      for sseed in 42 43 44; do
        if (( task_index % 2 == NODE_INDEX )); then
          gpu=$((local_index % GPU_COUNT))
          log="$LOG_ROOT/lambda_$(printf '%+d' "$lambda")_${supervision}_r${rseed}_s${sseed}.log"
          CUDA_VISIBLE_DEVICES="$gpu" bash "$ROOT/class_in_class/run_imagenette_entropy_high_lambda.sh" \
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

if [[ "$NODE_INDEX" == 0 ]]; then
  while true; do
    other="$(python -c "import json;print(json.load(open('$EXP_ROOT/status/high_lambda_node1.json')).get('status','missing'))" 2>/dev/null || echo missing)"
    [[ "$other" == complete ]] && break
    [[ "$other" == failed ]] && { echo "node1 high-lambda shard failed" >&2; exit 1; }
    sleep 60
  done
  mkdir -p "$EXP_ROOT/summary"
  python "$ROOT/class_in_class/summarize_imagenette_entropy_selection.py" \
      --experiment-root "$EXP_ROOT" --lambdas -4 -2 0 2 4 8 16 32 \
      --output "$EXP_ROOT/summary/extended_full.json" \
      >"$EXP_ROOT/logs/high_lambda_summary.log" 2>&1
fi
