#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_ROOT="${CODA_BASE_ROOT:-/linxi/dataset/FG_CoDA_standard/v2}"
LOCK_ROOT="$BASE_ROOT/locks"
STATUS_PATH="$BASE_ROOT/paired_queue_status.json"
SUMMARY_PATH="$BASE_ROOT/summary/coda_dino_minus_vae.json"
CURRENT_STAGE=initializing
mkdir -p "$LOCK_ROOT" "$BASE_ROOT/summary"

write_status() {
    local status="$1" stage="$2" exit_code="${3:-0}"
    python - "$STATUS_PATH" "$status" "$stage" "$exit_code" <<'PY'
import datetime, json, os, sys
from pathlib import Path
path=Path(sys.argv[1])
payload={
    'status':sys.argv[2],
    'stage':sys.argv[3],
    'exit_code':int(sys.argv[4]),
    'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'order':['vae_space','dino_space'],
}
tmp=path.with_suffix(path.suffix+'.tmp')
tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
os.replace(tmp,path)
PY
}

on_exit() {
    local exit_code=$?
    if (( exit_code != 0 )); then
        write_status failed "$CURRENT_STAGE" "$exit_code"
    fi
}
trap on_exit EXIT

# Hold the same lock across both branches so no manual launcher can enter the
# small gap between VAE completion and DINO startup.
exec 8>"$LOCK_ROOT/launcher.lock"
flock -n 8 || { echo "Another CoDA queue is already running" >&2; exit 1; }

write_status running vae_space
CURRENT_STAGE=vae_space
CODA_EXTERNAL_LOCK_HELD=1 bash "$ROOT_DIR/CV-DD/fine_grained/coda_fg_queue.sh" vae_space

write_status running dino_space
CURRENT_STAGE=dino_space
CODA_EXTERNAL_LOCK_HELD=1 bash "$ROOT_DIR/CV-DD/fine_grained/coda_fg_queue.sh" dino_space

CURRENT_STAGE=paired_summary
python "$ROOT_DIR/CV-DD/fine_grained/summarize_coda_space_comparison.py" \
    --base-root "$BASE_ROOT" --output "$SUMMARY_PATH"

CURRENT_STAGE=complete
write_status complete complete
trap - EXIT
