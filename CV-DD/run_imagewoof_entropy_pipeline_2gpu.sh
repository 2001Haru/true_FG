#!/usr/bin/env bash
set -euo pipefail

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}"
export ENTROPY_DATASET_PROFILE=imagewoof
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${IMAGEWOOF_DATA_ROOT:-/linxi/dataset/CV-DD/test_data/imagewoof}"
MASTER_ROOT="${IMAGEWOOF_MASTER_ROOT:-/linxi/dataset/CV-DD/experiments/imagewoof_cvdd_v1}"
TEACHER_ROOT="$MASTER_ROOT/teacher/workers8_eval_sparse_run2"
TEACHER="$TEACHER_ROOT/ResNet18.pth"
EXP_ROOT="${IMAGEWOOF_ENTROPY_ROOT:-/linxi/dataset/CV-DD/experiments/imagewoof_entropy_selection_v1}"
DINO_MODEL="${DINOV2_MODEL_ROOT:-/linxi/models/DINOv2/dinov2-base}"
DINO_CACHE="$EXP_ROOT/cache/imagewoof_dinov2.pt"
STATUS="$MASTER_ROOT/pipeline_status.json"
LOG_ROOT="$MASTER_ROOT/logs"
mkdir -p "$LOG_ROOT" "$EXP_ROOT/logs" "$EXP_ROOT/status" "$EXP_ROOT/locks"

write_status(){ python - "$STATUS" "$1" "$2" <<'PY'
import datetime,json,os,sys
from pathlib import Path
p=Path(sys.argv[1]);d={'status':sys.argv[2],'stage':sys.argv[3],'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()};t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');os.replace(t,p)
PY
}
on_exit(){ code=$?; ((code==0)) || write_status failed "${CURRENT_STAGE:-unknown}:exit${code}"; }
trap on_exit EXIT
exec 9>"$MASTER_ROOT/pipeline.lock"
flock -n 9 || { echo "ImageWoof pipeline already active" >&2; exit 1; }

CURRENT_STAGE=teacher
write_status running "$CURRENT_STAGE"
if [[ ! -f "$TEACHER" || ! -f "$TEACHER_ROOT/training_history.json" ]]; then
    [[ ! -e "$TEACHER_ROOT" ]] || { echo "partial Teacher output exists: $TEACHER_ROOT" >&2; exit 1; }
    CUDA_VISIBLE_DEVICES=0 python -u "$ROOT/class_in_class/train_imagewoof_teacher_cvdd.py" \
        --data-dir "$DATA_ROOT" --output-dir "$TEACHER_ROOT" --workers 8 --test-every 10 \
        >"$LOG_ROOT/teacher_workers8_eval_sparse_run2.log" 2>&1
fi
python - "$TEACHER_ROOT/training_history.json" "$TEACHER" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=json.load(open(sys.argv[1]));c=Path(sys.argv[2]);h=hashlib.sha256(c.read_bytes()).hexdigest()
assert p['status']=='complete' and p['epochs']==300 and p['workers']==8 and p['evaluation_count']==36
assert p['checkpoint_selection']=='final epoch only' and p['checkpoint_sha256']==h
assert p['train_images']==9025 and p['test_images']==3929
PY

CURRENT_STAGE=dino
write_status running "$CURRENT_STAGE"
CUDA_VISIBLE_DEVICES=0 python -u "$ROOT/class_in_class/encode_imagenette_dinov2.py" \
    --data-root "$DATA_ROOT" --model-dir "$DINO_MODEL" --output "$DINO_CACHE" \
    --splits train test --batch-size 128 --workers 8 \
    >"$EXP_ROOT/logs/dino.log" 2>&1

CURRENT_STAGE=selection_preflight
write_status running "$CURRENT_STAGE"
if [[ ! -f "$EXP_ROOT/preflight/selection_preflight.json" ]]; then
    CUDA_VISIBLE_DEVICES=0 python -u "$ROOT/class_in_class/prepare_imagenette_entropy_selection.py" \
        --data-root "$DATA_ROOT" --teacher-checkpoint "$TEACHER" --dino-cache "$DINO_CACHE" \
        --output-root "$EXP_ROOT" --batch-size 512 --workers 12 \
        >"$EXP_ROOT/logs/selection_preflight.log" 2>&1
fi
python "$ROOT/class_in_class/approve_imagewoof_entropy_gate.py" --phase base \
    --experiment-root "$EXP_ROOT" --teacher "$TEACHER" \
    >"$EXP_ROOT/logs/base_gate.log" 2>&1
if [[ ! -f "$EXP_ROOT/preflight/high_lambda_extension_audit.json" ]]; then
    python -u "$ROOT/class_in_class/prepare_imagenette_entropy_high_lambda.py" \
        --data-root "$DATA_ROOT" --teacher-checkpoint "$TEACHER" --dino-cache "$DINO_CACHE" \
        --experiment-root "$EXP_ROOT" >"$EXP_ROOT/logs/high_lambda_preflight.log" 2>&1
fi
python "$ROOT/class_in_class/approve_imagewoof_entropy_gate.py" --phase high \
    --experiment-root "$EXP_ROOT" --teacher "$TEACHER" \
    >"$EXP_ROOT/logs/high_gate.log" 2>&1

export IMAGEWOOF_DATA_ROOT="$DATA_ROOT" IMAGEWOOF_ENTROPY_ROOT="$EXP_ROOT" IMAGEWOOF_TEACHER="$TEACHER"
export ENTROPY_EVAL_WORKERS="${ENTROPY_EVAL_WORKERS:-4}"
wait_wave(){ local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; ((failed==0)); }
launch_wave(){
    local -n tasks_ref=$1
    local pids=() index=0 task gpu log
    for task in "${tasks_ref[@]}"; do
        read -r arm rseed sseed supervision <<<"$task"
        gpu=$((index%2)); log="$EXP_ROOT/logs/scan/${arm}_${supervision}_r${rseed}_s${sseed}.log"
        mkdir -p "$(dirname "$log")"
        CUDA_VISIBLE_DEVICES="$gpu" bash "$ROOT/class_in_class/run_imagewoof_entropy_one.sh" \
            "$arm" "$rseed" "$sseed" "$supervision" >"$log" 2>&1 &
        pids+=("$!"); index=$((index+1))
        if ((${#pids[@]}==4)); then wait_wave "${pids[@]}"; pids=(); fi
    done
    if ((${#pids[@]}>0)); then wait_wave "${pids[@]}"; fi
}

CURRENT_STAGE=lambda_scan
write_status running "$CURRENT_STAGE"
tasks=()
for arm in -4 -2 0 2 4 8 16 32; do
    for supervision in hard soft1; do
        for rseed in 0 1 2; do
            for sseed in 42 43 44; do tasks+=("$arm $rseed $sseed $supervision"); done
        done
    done
done
launch_wave tasks

CURRENT_STAGE=top10_preflight
write_status running "$CURRENT_STAGE"
if [[ ! -f "$EXP_ROOT/preflight/lambda0_entropy_allocation_audit.json" ]]; then
    python -u "$ROOT/class_in_class/prepare_imagenette_entropy_allocation.py" \
        --data-root "$DATA_ROOT" --experiment-root "$EXP_ROOT" \
        >"$EXP_ROOT/logs/top10_preflight.log" 2>&1
fi
python "$ROOT/class_in_class/approve_imagewoof_entropy_gate.py" --phase top10 \
    --experiment-root "$EXP_ROOT" --teacher "$TEACHER" \
    >"$EXP_ROOT/logs/top10_gate.log" 2>&1

CURRENT_STAGE=top10
write_status running "$CURRENT_STAGE"
tasks=()
for supervision in hard soft1; do for sseed in 42 43 44; do tasks+=("top10 0 $sseed $supervision"); done; done
launch_wave tasks

CURRENT_STAGE=summary
write_status running "$CURRENT_STAGE"
python "$ROOT/class_in_class/summarize_imagewoof_entropy_scan.py" \
    --experiment-root "$EXP_ROOT" --output "$EXP_ROOT/summary/complete_scan.json" \
    >"$EXP_ROOT/logs/summary.log" 2>&1
write_status complete complete
trap - EXIT
