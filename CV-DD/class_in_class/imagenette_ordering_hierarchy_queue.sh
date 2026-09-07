#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; NODE="${1:?node shard 0 or 1}"; [[ "$NODE" == 0 || "$NODE" == 1 ]] || exit 2
EXP_ROOT="${IMAGENETTE_ORDERING_ROOT:-/linxi/dataset/CV-DD/experiments/imagenette_ordering_hierarchy_v1}"; LOG="$EXP_ROOT/logs/node${NODE}"; STATUS="$EXP_ROOT/status/node${NODE}.json"; mkdir -p "$LOG" "$(dirname "$STATUS")" "$EXP_ROOT/locks"
write(){ python - "$STATUS" "$1" "${2:-0}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
p=Path(sys.argv[1]);x={'status':sys.argv[2],'exit_code':int(sys.argv[3]),'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()};t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
PY
}
exec 9>"$EXP_ROOT/locks/node${NODE}.lock";flock -n 9||exit 1;trap 'c=$?;((c==0))||write failed "$c"' EXIT;write running
waitwave(){ local f=0 p;for p in "$@";do wait "$p"||f=1;done;((f==0));}; idx=0;localidx=0;pids=()
pair(){ case "$1" in 3|6|9) echo '42 43';;4|7|10) echo '43 44';;5|8|11) echo '42 44';;esac; }
for r in {3..11};do for s in $(pair "$r");do
 if ((idx%2==NODE));then
  # A tuple is launched as an indivisible four-arm group on one GPU.
  gpu=$((localidx%2)); log="$LOG/r${r}_s${s}.log"; (for arm in O B A Aprime;do CUDA_VISIBLE_DEVICES="$gpu" bash "$ROOT/class_in_class/run_imagenette_ordering_hierarchy.sh" "$arm" "$r" "$s";done)>"$log" 2>&1 & pids+=("$!");localidx=$((localidx+1));
  if ((${#pids[@]}==4));then waitwave "${pids[@]}";pids=();fi
 fi;idx=$((idx+1));
done;done
if ((${#pids[@]}>0));then waitwave "${pids[@]}";fi;write complete;trap - EXIT
if [[ "$NODE" == 0 ]];then
 while true;do x=$(python -c "import json;print(json.load(open('$EXP_ROOT/status/node1.json'))['status'])" 2>/dev/null||echo missing);[[ "$x" == complete ]]&&break;[[ "$x" == failed ]]&&exit 1;sleep 60;done
 python "$ROOT/class_in_class/summarize_imagenette_ordering_hierarchy.py" --experiment-root "$EXP_ROOT" --output "$EXP_ROOT/summary/ordering_hierarchy.json" >"$EXP_ROOT/logs/summary.log" 2>&1
fi
