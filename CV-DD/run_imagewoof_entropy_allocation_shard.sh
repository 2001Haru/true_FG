#!/usr/bin/env bash
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export ENTROPY_DATASET_PROFILE=imagewoof
NODE_INDEX="${1:?usage: run_imagewoof_entropy_allocation_shard.sh 0|1}"
[[ "$NODE_INDEX" == 0 || "$NODE_INDEX" == 1 ]] || { echo "node index must be 0 or 1" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="${IMAGEWOOF_ENTROPY_ROOT:-/linxi/dataset/CV-DD/experiments/imagewoof_entropy_selection_v1}"
STATUS="$EXP_ROOT/status/allocation_node${NODE_INDEX}.json"
LOG_ROOT="$EXP_ROOT/logs/allocation_node${NODE_INDEX}"
mkdir -p "$LOG_ROOT" "$EXP_ROOT/status" "$EXP_ROOT/locks"

write_status(){ python - "$STATUS" "$1" "$2" <<'PY'
import datetime,json,os,sys
from pathlib import Path
p=Path(sys.argv[1]);d={'status':sys.argv[2],'stage':sys.argv[3],'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()};t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');os.replace(t,p)
PY
}
on_exit(){ code=$?; ((code==0)) || write_status failed "training:exit${code}"; }
trap on_exit EXIT
exec 9>"$EXP_ROOT/locks/allocation_node${NODE_INDEX}.lock"
flock -n 9 || { echo "allocation shard already active" >&2; exit 1; }
write_status running training

wait_wave(){ local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; ((failed==0)); }
pids=(); local_index=0; task_index=0
for arm in high_entropy_soft1 low_entropy_soft1; do
  for rseed in 0 1 2; do
    for sseed in 42 43 44; do
      if ((task_index%2==NODE_INDEX)); then
        result="$EXP_ROOT/results/lambda0_entropy_allocation/rseed${rseed}/${arm}_sseed${sseed}.json"
        if [[ ! -f "$result" ]]; then
          gpu=$((local_index%2)); log="$LOG_ROOT/${arm}_r${rseed}_s${sseed}.log"
          CUDA_VISIBLE_DEVICES="$gpu" bash "$ROOT/class_in_class/run_imagewoof_entropy_allocation_one.sh" \
              "$arm" "$rseed" "$sseed" >"$log" 2>&1 &
          pids+=("$!"); local_index=$((local_index+1))
          if ((${#pids[@]}==4)); then wait_wave "${pids[@]}"; pids=(); fi
        fi
      fi
      task_index=$((task_index+1))
    done
  done
done
if ((${#pids[@]}>0)); then wait_wave "${pids[@]}"; fi
write_status running training_complete

peer="$EXP_ROOT/status/allocation_node$((1-NODE_INDEX)).json"
while true; do
  value="$(python - "$peer" <<'PY'
import json,sys
try:
 d=json.load(open(sys.argv[1]));print('failed' if d.get('status')=='failed' else ('ready' if d.get('stage') in ('training_complete','summary','complete') else 'wait'))
except Exception:print('wait')
PY
)"
  [[ "$value" == ready ]] && break
  [[ "$value" == failed ]] && { echo "peer allocation shard failed" >&2; exit 1; }
  sleep 30
done
if [[ "$NODE_INDEX" == 0 ]]; then
  write_status running summary
  python "$ROOT/class_in_class/summarize_imagenette_entropy_allocation.py" \
      --experiment-root "$EXP_ROOT" --experiment-name imagewoof_entropy_allocation_v1 \
      --output "$EXP_ROOT/summary/entropy_allocation.json" >"$LOG_ROOT/summary.log" 2>&1
fi
write_status complete complete
trap - EXIT
