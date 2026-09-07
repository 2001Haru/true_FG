#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${IMAGENETTE_ENTROPY_ROOT:-/linxi/dataset/CV-DD/experiments/imagenette_entropy_selection_v1}"
LOG_ROOT="$EXP_ROOT/logs/nontarget_permutation"
STATUS="$EXP_ROOT/status/nontarget_permutation_node30832.json"
GPU_COUNT="${ENTROPY_GPU_COUNT:-2}"
EVALS_PER_GPU="${ENTROPY_EVALS_PER_GPU:-2}"
WAVE_SIZE=$((GPU_COUNT * EVALS_PER_GPU))
mkdir -p "$LOG_ROOT" "$(dirname "$STATUS")" "$EXP_ROOT/locks"

write_status() {
  python - "$STATUS" "$1" "${2:-0}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); x={'status':sys.argv[2],'exit_code':int(sys.argv[3]),'node':'30832-only','updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}; t=p.with_suffix('.tmp'); t.write_text(json.dumps(x,indent=2)+'\n'); os.replace(t,p)
PY
}
on_exit(){ code=$?; (( code == 0 )) || write_status failed "$code"; }
exec 9>"$EXP_ROOT/locks/nontarget_permutation_node30832.lock"
flock -n 9 || { echo "permutation queue already active" >&2; exit 1; }
trap on_exit EXIT
write_status running 0
wait_wave(){ local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; (( failed == 0 )); }

local_index=0; pids=()
for group in high low; do
  for rseed in 0 1 2; do
    for sseed in 42 43 44; do
      gpu=$((local_index % GPU_COUNT))
      log="$LOG_ROOT/${group}_r${rseed}_s${sseed}.log"
      CUDA_VISIBLE_DEVICES="$gpu" bash "$ROOT/class_in_class/run_imagenette_nontarget_permutation.sh" \
        "$group" "$rseed" "$sseed" >"$log" 2>&1 &
      pids+=("$!"); local_index=$((local_index+1))
      if (( ${#pids[@]} == WAVE_SIZE )); then wait_wave "${pids[@]}"; pids=(); fi
    done
  done
done
if (( ${#pids[@]} > 0 )); then wait_wave "${pids[@]}"; fi
python "$ROOT/class_in_class/summarize_imagenette_nontarget_permutation.py" \
  --experiment-root "$EXP_ROOT" --output "$EXP_ROOT/summary/nontarget_permutation.json" \
  >"$LOG_ROOT/summary.log" 2>&1
write_status complete 0
trap - EXIT

