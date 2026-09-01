#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STANDARD_ROOT="${STANDARD_ROOT:-/linxi/dataset/FG_SRe2L_standard/v1}"
SENSITIVITY_ROOT="${SENSITIVITY_ROOT:-$STANDARD_ROOT/sensitivity/cub_iter4000_t42_r42_s42}"
PIPELINE_ROOT="$SENSITIVITY_ROOT/pipeline"
PREPARED_DATA_ROOT="${PREPARED_DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
TEACHER_DIR="$STANDARD_ROOT/teachers/CUB_imsize224/tseed42"
PATCH_BASE="$STANDARD_ROOT/arms/tseed42/patches/CUB_imsize224/tseed42_pseed42"
LOG_ROOT="$SENSITIVITY_ROOT/logs"
STATUS_ROOT="$SENSITIVITY_ROOT/status"
RESULT_ROOT="$SENSITIVITY_ROOT/results"
POST_EVAL_ROOT="$SENSITIVITY_ROOT/post_eval"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT"

cleanup() {
    local rc=$?
    if (( rc != 0 )); then
        echo "$(date --iso-8601=seconds) sensitivity failed exit=$rc" > "$STATUS_ROOT/launcher.failed"
    fi
    rm -f "$STATUS_ROOT/launcher.running"
    return "$rc"
}
trap cleanup EXIT

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export TORCH_HOME="${TORCH_HOME:-/linxi/dataset/FD2/torch_cache}"
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"
export EXP_ROOT="$PIPELINE_ROOT"
export PREPARED_DATA_ROOT
export TEACHER_DIR_OVERRIDE="$TEACHER_DIR"
export TEACHER_SEED=42
export PATCH_SEED=42
export PATCH_BASE_OVERRIDE="$PATCH_BASE"
export RECOVERY_ITERATIONS_OVERRIDE=4000
export RELABEL_WORKERS=8
export RELABEL_MANIFEST_REQUIRED=1
export EVAL_WORKERS=8
export EVAL_PERSISTENT_WORKERS=1
export STUDENT_INITIALIZATION=random
export STUDENT_ADAMW_LR=1e-3
export STUDENT_ADAMW_WEIGHT_DECAY=1e-5
export STUDENT_ETA=2
export STUDENT_TEMPERATURE=20
export RESULT_ROOT POST_EVAL_ROOT

exec 9>"$STATUS_ROOT/launcher.lock"
flock -n 9 || { echo "CUB iteration sensitivity already running" >&2; exit 1; }
[[ -f "$STANDARD_ROOT/summary/completion_audit.json" ]] || {
    echo "standard completion audit is missing" >&2; exit 1;
}
[[ -f "$TEACHER_DIR/ResNet18.pth" && -d "$PATCH_BASE/2" ]] || {
    echo "paired Teacher or patch tree is missing" >&2; exit 1;
}

python - "$SENSITIVITY_ROOT" "$STANDARD_ROOT" "$TEACHER_DIR" "$PATCH_BASE" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

root, standard, teacher, patch = map(Path, sys.argv[1:])
payload = {
    'status': 'running',
    'experiment': 'cub_recovery_iterations_4000_vs_10000',
    'single_changed_variable': 'recovery_iterations: 10000 -> 4000',
    'dataset': 'CUB_imsize224',
    'teacher_seed': 42,
    'recovery_seed': 42,
    'student_seed': 42,
    'ipcs': [1, 3, 5],
    'teacher_checkpoint': str((teacher / 'ResNet18.pth').resolve()),
    'teacher_checkpoint_sha256': sha256(teacher / 'ResNet18.pth'),
    'patch_tree': str((patch / '2').resolve()),
    'standard_root': str(standard.resolve()),
    'sensitivity_root': str(root.resolve()),
}
path = root / 'experiment_definition.json'
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix('.json.tmp')
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
os.replace(tmp, path)
PY

echo "$(date --iso-8601=seconds) sensitivity started" > "$STATUS_ROOT/launcher.running"
CUDA_VISIBLE_DEVICES=0 bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" recover \
    CUB_imsize224 42 > "$LOG_ROOT/recover.log" 2>&1
bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" sample \
    CUB_imsize224 42 > "$LOG_ROOT/sample.log" 2>&1
for ipc in 1 3 5; do
    CUDA_VISIBLE_DEVICES=0 bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" relabel \
        CUB_imsize224 42 "$ipc" > "$LOG_ROOT/relabel_ipc${ipc}.log" 2>&1
done

run_eval() {
    local gpu="$1" ipc="$2"
    CUDA_VISIBLE_DEVICES="$gpu" bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" eval \
        CUB_imsize224 42 "$ipc" 42 > "$LOG_ROOT/eval_ipc${ipc}_sseed42.log" 2>&1
}

run_eval 0 1 & pid1=$!
run_eval 0 3 & pid3=$!
run_eval 1 5 & pid5=$!
failed=0
wait "$pid1" || failed=1
wait "$pid3" || failed=1
wait "$pid5" || failed=1
(( failed == 0 )) || { echo "post-evaluation failed" >&2; exit 1; }

python "$ROOT_DIR/fine_grained/summarize_cub_iteration_sensitivity.py" \
    --sensitivity-root "$SENSITIVITY_ROOT" --standard-root "$STANDARD_ROOT" \
    > "$LOG_ROOT/summary.log" 2>&1
rm -f "$STATUS_ROOT/launcher.running"
echo "$(date --iso-8601=seconds) sensitivity complete" > "$STATUS_ROOT/launcher.complete"
trap - EXIT
