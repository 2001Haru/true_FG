#!/usr/bin/env bash
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export ENTROPY_DATASET_PROFILE=imagewoof
NODE_INDEX="${1:?usage: run_imagewoof_entropy_remaining_shard.sh 0|1}"
[[ "$NODE_INDEX" == 0 || "$NODE_INDEX" == 1 ]] || { echo "node index must be 0 or 1" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${IMAGEWOOF_DATA_ROOT:-/linxi/dataset/CV-DD/test_data/imagewoof}"
MASTER_ROOT="${IMAGEWOOF_MASTER_ROOT:-/linxi/dataset/CV-DD/experiments/imagewoof_cvdd_v1}"
TEACHER="${IMAGEWOOF_TEACHER:-$MASTER_ROOT/teacher/workers8_eval_sparse_run2/ResNet18.pth}"
EXP_ROOT="${IMAGEWOOF_ENTROPY_ROOT:-/linxi/dataset/CV-DD/experiments/imagewoof_entropy_selection_v1}"
STATUS="$EXP_ROOT/status/remaining_node${NODE_INDEX}.json"
LOG_ROOT="$EXP_ROOT/logs/remaining_node${NODE_INDEX}"
mkdir -p "$LOG_ROOT" "$EXP_ROOT/status" "$EXP_ROOT/locks"
export IMAGEWOOF_DATA_ROOT="$DATA_ROOT" IMAGEWOOF_ENTROPY_ROOT="$EXP_ROOT" IMAGEWOOF_TEACHER="$TEACHER"
export ENTROPY_EVAL_WORKERS="${ENTROPY_EVAL_WORKERS:-4}"

write_status(){ python - "$STATUS" "$1" "$2" <<'PY'
import datetime,json,os,sys
from pathlib import Path
p=Path(sys.argv[1]);d={'status':sys.argv[2],'stage':sys.argv[3],'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()};t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');os.replace(t,p)
PY
}
CURRENT_STAGE=start
on_exit(){ code=$?; ((code==0)) || write_status failed "${CURRENT_STAGE}:exit${code}"; }
trap on_exit EXIT
exec 9>"$EXP_ROOT/locks/remaining_node${NODE_INDEX}.lock"
flock -n 9 || { echo "remaining shard already active: node${NODE_INDEX}" >&2; exit 1; }

result_path(){
    local arm="$1" rseed="$2" sseed="$3" supervision="$4"
    if [[ "$arm" == top10 ]]; then
        echo "$EXP_ROOT/results/highest_entropy_top10/${supervision}_sseed${sseed}.json"
    else
        echo "$EXP_ROOT/results/lambda_$(printf '%+d' "$arm")/rseed${rseed}/${supervision}_sseed${sseed}.json"
    fi
}
wait_wave(){ local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; ((failed==0)); }
run_tasks(){
    local phase="$1"; shift
    local tasks=("$@") pids=() local_index=0 task arm rseed sseed supervision result gpu log
    for task in "${tasks[@]}"; do
        read -r arm rseed sseed supervision <<<"$task"
        result="$(result_path "$arm" "$rseed" "$sseed" "$supervision")"
        [[ -f "$result" ]] && continue
        gpu=$((local_index%2)); log="$LOG_ROOT/${phase}_${arm}_${supervision}_r${rseed}_s${sseed}.log"
        CUDA_VISIBLE_DEVICES="$gpu" bash "$ROOT/class_in_class/run_imagewoof_entropy_one.sh" \
            "$arm" "$rseed" "$sseed" "$supervision" >"$log" 2>&1 &
        pids+=("$!"); local_index=$((local_index+1))
        if ((${#pids[@]}==4)); then wait_wave "${pids[@]}"; pids=(); fi
    done
    if ((${#pids[@]}>0)); then wait_wave "${pids[@]}"; fi
}
wait_peer(){
    local stage="$1" peer=$((1-NODE_INDEX)) peer_status="$EXP_ROOT/status/remaining_node$((1-NODE_INDEX)).json"
    while true; do
        value="$(python - "$peer_status" "$stage" <<'PY'
import json,sys
try:
 d=json.load(open(sys.argv[1]));current=d.get('stage');wanted=sys.argv[2]
 ready={
  'numeric_complete':{'numeric_complete','top10_preflight','top10','top10_complete','summary','complete'},
  'top10_complete':{'top10_complete','summary','complete'},
 }[wanted]
 print('failed' if d.get('status')=='failed' else ('ready' if current in ready else 'wait'))
except Exception:print('wait')
PY
)"
        [[ "$value" == ready ]] && return
        [[ "$value" == failed ]] && { echo "peer shard failed" >&2; exit 1; }
        sleep 30
    done
}

for required in "$DATA_ROOT/train" "$DATA_ROOT/test" "$TEACHER" \
    "$EXP_ROOT/preflight/manual_visual_gate.json" "$EXP_ROOT/preflight/high_lambda_extension_gate.json"; do
    [[ -e "$required" ]] || { echo "missing required input: $required" >&2; exit 1; }
done

CURRENT_STAGE=lambda_scan
write_status running "$CURRENT_STAGE"
tasks=(); task_index=0
for arm in -4 -2 0 2 4 8 16 32; do
  for supervision in hard soft1; do
    for rseed in 0 1 2; do
      for sseed in 42 43 44; do
        if ((task_index%2==NODE_INDEX)); then tasks+=("$arm $rseed $sseed $supervision"); fi
        task_index=$((task_index+1))
      done
    done
  done
done
run_tasks lambda "${tasks[@]}"
write_status running numeric_complete
wait_peer numeric_complete

CURRENT_STAGE=top10_preflight
if [[ "$NODE_INDEX" == 0 ]]; then
    if [[ ! -f "$EXP_ROOT/preflight/lambda0_entropy_allocation_audit.json" ]]; then
        python -u "$ROOT/class_in_class/prepare_imagenette_entropy_allocation.py" \
            --data-root "$DATA_ROOT" --experiment-root "$EXP_ROOT" >"$LOG_ROOT/top10_preflight.log" 2>&1
    fi
    python "$ROOT/class_in_class/approve_imagewoof_entropy_gate.py" --phase top10 \
        --experiment-root "$EXP_ROOT" --teacher "$TEACHER" >"$LOG_ROOT/top10_gate.log" 2>&1
else
    while [[ ! -f "$EXP_ROOT/preflight/lambda0_entropy_allocation_gate.json" ]]; do sleep 15; done
fi

CURRENT_STAGE=top10
write_status running "$CURRENT_STAGE"
tasks=(); task_index=0
for supervision in hard soft1; do
  for sseed in 42 43 44; do
    if ((task_index%2==NODE_INDEX)); then tasks+=("top10 0 $sseed $supervision"); fi
    task_index=$((task_index+1))
  done
done
run_tasks top10 "${tasks[@]}"
write_status running top10_complete
wait_peer top10_complete

CURRENT_STAGE=summary
if [[ "$NODE_INDEX" == 0 ]]; then
    python "$ROOT/class_in_class/summarize_imagewoof_entropy_scan.py" \
        --experiment-root "$EXP_ROOT" --output "$EXP_ROOT/summary/complete_scan.json" >"$LOG_ROOT/summary.log" 2>&1
fi
write_status complete complete
trap - EXIT
