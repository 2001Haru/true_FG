#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."&&pwd)"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_TeacherHard_RRC_v2/aircraft_ipc3_v1}"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
FKD_ROOT=/linxi/dataset/FG_HardAugFactorial_v2/aircraft_ipc3_v1
SOFT_ROOT=/linxi/dataset/FG_SoftAugFactorial_v2/aircraft_ipc3_v1
HARD_ROOT=/linxi/dataset/FG_HardAugFactorial_v2/aircraft_ipc3_v1
RANDOM_ROOT=/linxi/dataset/FG_RandomReal_soft_v2/aircraft_v1
RDED_ROOT=/linxi/dataset/FG_RDED_original_v2/aircraft_v1
GPUS=(${GPUS:-0 1}); EVALS_PER_GPU=${EVALS_PER_GPU:-3}
mkdir -p "$EXP_ROOT"/{logs,status,locks,results,audits,summary,post_eval}
ts(){ date --iso-8601=seconds; }
images(){ local method=$1; if [[ $method == random_real ]];then python - "$RANDOM_ROOT/results/A_imsize224/rseed0/ipc3_sseed42.json" <<'PY'
import json,sys;print(json.load(open(sys.argv[1]))['synthetic_data_path'])
PY
else python - "$RDED_ROOT/results/A_imsize224/gseed42/ipc3_sseed42.json" <<'PY'
import json,sys;print(json.load(open(sys.argv[1]))['synthetic_data_path'])
PY
fi; }
fkd(){ local method=$1 seed; [[ $method == random_real ]]&&seed=0||seed=42; echo "$FKD_ROOT/fkd/rrc_no_cutmix/$method/source_seed$seed/ipc3_bs20_ipc3"; }
audit_one(){ local method=$1; python "$ROOT_DIR/CV-DD/fine_grained/audit_teacher_hard_fkd.py" --images "$(images "$method")" --fkd "$(fkd "$method")" --output "$EXP_ROOT/audits/${method}.json"; }
eval_one(){ local method=$1 student=$2 gpu=$3 source_seed; [[ $method == random_real ]]&&source_seed=0||source_seed=42
 local result="$EXP_ROOT/results/$method/source_seed$source_seed/ipc3_sseed$student.json" log="$EXP_ROOT/logs/eval_${method}_s$student.log"; mkdir -p "$(dirname "$result")"
 CUDA_VISIBLE_DEVICES=$gpu PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u "$ROOT_DIR/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 --exp-name "teacher_hard_rrc_${method}_s$student" --original-data-path "$(images "$method")" --fkd-path "$(fkd "$method")" --output-dir "$EXP_ROOT/post_eval/$method/sseed$student" --batch-size 20 --epochs 400 --dataset-name A_imsize224 --gradient-accumulation-steps 2 --fkd-teacher-hard-label --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$student" --temperature 20 --student-initialization imagenet-v1 --student-protocol-name standard_protocol_v2_teacher_hard --adamw-weight-decay 1e-5 --adamw-beta1 .9 --adamw-beta2 .999 --adamw-eps 1e-8 --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 --val-dir "$DATA_ROOT/A_imsize224/test" --disable-wandb --per-class-output "$result" >"$log" 2>&1
}
main(){ exec 9>"$EXP_ROOT/locks/launcher.lock"; flock -n 9||exit 75; rm -f "$EXP_ROOT/status/launcher.failed" "$EXP_ROOT/status/launcher.complete"; echo "$(ts) started expected=6" >"$EXP_ROOT/status/launcher.running"
 audit_one random_real >"$EXP_ROOT/logs/audit_random_real.log" 2>&1 & a=$!; audit_one original_rded >"$EXP_ROOT/logs/audit_original_rded.log" 2>&1 & b=$!; wait "$a"; wait "$b"; echo "$(ts) complete" >"$EXP_ROOT/status/audit.complete"
 local pids=() i=0 method student gpu; for method in random_real original_rded;do for student in 42 43 44;do gpu=${GPUS[$((i%2))]}; eval_one "$method" "$student" "$gpu" & pids+=("$!"); i=$((i+1));done;done; local failed=0 p;for p in "${pids[@]}";do wait "$p"||failed=1;done;((failed==0))
 python "$ROOT_DIR/CV-DD/fine_grained/summarize_aircraft_teacher_hard_rrc.py" --root "$EXP_ROOT" --soft-root "$SOFT_ROOT" --hard-root "$HARD_ROOT" >"$EXP_ROOT/logs/summary.log" 2>&1
 rm -f "$EXP_ROOT/status/launcher.running"; echo "$(ts) complete" >"$EXP_ROOT/status/launcher.complete"; }
trap 'r=$?;if((r));then rm -f "$EXP_ROOT/status/launcher.running";echo "$(ts) exit=$r">"$EXP_ROOT/status/launcher.failed";fi' EXIT
main
