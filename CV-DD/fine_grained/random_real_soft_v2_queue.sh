#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard}"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_RandomReal_soft_v2/aircraft_v1}"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
TEACHER_DIR="${TEACHER_DIR:-/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42}"
EVALS_PER_GPU="${EVALS_PER_GPU:-4}"
LOG_ROOT="$EXP_ROOT/logs"; STATUS_ROOT="$EXP_ROOT/status"; LOCK_ROOT="$EXP_ROOT/locks"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$LOCK_ROOT" "$EXP_ROOT/summary"

write_definition(){
python - "$EXP_ROOT" "$SOURCE_ROOT" "$TEACHER_DIR" "$(git -C "$ROOT_DIR" rev-parse HEAD)" <<'PY'
import json,os,sys
from pathlib import Path
root,source,teacher=map(Path,sys.argv[1:4]); revision=sys.argv[4]
expected=[str((root/"results/A_imsize224"/f"rseed{r}"/f"ipc{i}_sseed{s}.json").resolve()) for i in (1,3,5) for r in (0,1,2) for s in (42,43,44)]
x={"status":"running","experiment":"aircraft_random_real_fixed_teacher_soft_v2","git_revision":revision,
"dataset":"A_imsize224","ipcs":[1,3,5],"selection_seeds":[0,1,2],"student_seeds":[42,43,44],
"teacher_seed":42,"teacher_dir":str(teacher.resolve()),"selection_source":str(source.resolve()),"expected_fkd_sets":9,
"expected_results":27,"supervision":"SRe2L++ Teacher42 BSSL/FKD soft labels","student_protocol":"v2","expected_result_files":expected}
p=root/"matrix_definition.json"; t=p.with_suffix(".json.tmp"); t.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n"); os.replace(t,p)
PY
}

relabel_one(){
    local ipc="$1" selection="$2" gpu="$3"
    local selected="$SOURCE_ROOT/selected/A_imsize224/rseed${selection}/ipc${ipc}"
    local manifest="$SOURCE_ROOT/manifests/A_imsize224/rseed${selection}/ipc${ipc}.json"
    local base="$EXP_ROOT/fkd/A_imsize224/rseed${selection}/ipc${ipc}"
    local actual="${base}_bs20_ipc${ipc}" expected=$((400*100*ipc/20)) count=0
    [[ -f "$manifest" && -d "$selected" && -f "$TEACHER_DIR/ResNet18.pth" ]]
    [[ -d "$actual" ]] && count="$(find "$actual" -type f -name 'batch_*.tar' | wc -l)"
    if (( count != expected )); then
        CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
        python -u "$ROOT_DIR/CV-DD/relabel/relabel.py" --syn-data-path "$selected" --fkd-path "$base" \
          --model-pool-dir "$TEACHER_DIR" --teacher-model-name ResNet18 --gpu 0 --batch-size 20 --workers 8 \
          --dataset-name A_imsize224 --epochs 400 --seed 42 --fkd-seed 42 --min-scale-crops 0.08 \
          --max-scale-crops 1 --use-fp16 --mode fkd_save --mix-type cutmix \
          > "$LOG_ROOT/relabel_ipc${ipc}_rseed${selection}.log" 2>&1
    fi
    python "$ROOT_DIR/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$actual" --images $((100*ipc)) \
      --classes 100 --batch-size 20 --epochs 400 >> "$LOG_ROOT/relabel_ipc${ipc}_rseed${selection}.log" 2>&1
}

eval_one(){
    local ipc="$1" selection="$2" student="$3" gpu="$4"
    local selected="$SOURCE_ROOT/selected/A_imsize224/rseed${selection}/ipc${ipc}"
    local manifest="$SOURCE_ROOT/manifests/A_imsize224/rseed${selection}/ipc${ipc}.json"
    local fkd="$EXP_ROOT/fkd/A_imsize224/rseed${selection}/ipc${ipc}_bs20_ipc${ipc}"
    local result="$EXP_ROOT/results/A_imsize224/rseed${selection}/ipc${ipc}_sseed${student}.json"
    local log="$LOG_ROOT/eval_ipc${ipc}_rseed${selection}_sseed${student}.log"
    mkdir -p "$(dirname "$result")"; exec 7>"${result}.lock"; flock -n 7 || return 75
    if [[ ! -f "$result" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -u "$ROOT_DIR/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc "$ipc" \
        --exp-name "random_real_soft_v2_A_ipc${ipc}_r${selection}_s${student}" --original-data-path "$selected" \
        --fkd-path "$fkd" --output-dir "$EXP_ROOT/post_eval" --batch-size 20 --epochs 400 \
        --dataset-name A_imsize224 --gradient-accumulation-steps 2 --mix-type cutmix --workers 8 \
        --persistent-workers --fkd_seed 42 --train-seed "$student" --temperature 20 \
        --student-initialization imagenet-v1 --student-protocol-name standard_protocol_v2 \
        --adamw-weight-decay 1e-5 --adamw-beta1 0.9 --adamw-beta2 0.999 --adamw-eps 1e-8 \
        --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 \
        --val-dir "$DATA_ROOT/A_imsize224/test" --disable-wandb --per-class-output "$result" > "$log" 2>&1
    fi
    python "$ROOT_DIR/CV-DD/fine_grained/record_random_real_soft_v2_result.py" --result "$result" \
      --selection-manifest "$manifest" --teacher-dir "$TEACHER_DIR" --fkd-dir "$fkd" --ipc "$ipc" \
      --selection-seed "$selection" --student-seed "$student" >> "$log" 2>&1
}

wait_all(){ local failed=0 p; for p in "$@"; do wait "$p" || failed=1; done; ((failed==0)); }

main(){
  exec 9>"$LOCK_ROOT/launcher.lock"; flock -n 9 || exit 75
  write_definition; echo "$(date --iso-8601=seconds) started" > "$STATUS_ROOT/launcher.running"
  local pids=() idx=0 ipc selection student gpu
  for ipc in 1 3 5; do for selection in 0 1 2; do
    gpu=$((idx%2)); relabel_one "$ipc" "$selection" "$gpu" & pids+=("$!"); idx=$((idx+1))
    if (( ${#pids[@]}==2 )); then if ! wait_all "${pids[@]}"; then return 1; fi; pids=(); fi
  done; done
  if (( ${#pids[@]} )); then if ! wait_all "${pids[@]}"; then return 1; fi; fi
  pids=(); idx=0
  for ipc in 1 3 5; do for selection in 0 1 2; do for student in 42 43 44; do
    gpu=$((idx%2)); eval_one "$ipc" "$selection" "$student" "$gpu" & pids+=("$!"); idx=$((idx+1))
    if (( ${#pids[@]}==2*EVALS_PER_GPU )); then if ! wait_all "${pids[@]}"; then return 1; fi; pids=(); fi
  done; done; done
  if (( ${#pids[@]} )); then if ! wait_all "${pids[@]}"; then return 1; fi; fi
  python "$ROOT_DIR/CV-DD/fine_grained/summarize_random_real_soft_v2.py" --root "$EXP_ROOT" > "$LOG_ROOT/summary.log" 2>&1
  python - "$EXP_ROOT/matrix_definition.json" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads(p.read_text()); x["status"]="complete"; t=p.with_suffix(".json.tmp"); t.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n"); os.replace(t,p)
PY
  rm -f "$STATUS_ROOT/launcher.running"; echo "$(date --iso-8601=seconds) complete" > "$STATUS_ROOT/launcher.complete"
}
trap 'rc=$?; if ((rc)); then rm -f "$STATUS_ROOT/launcher.running"; echo "$(date --iso-8601=seconds) exit=$rc" > "$STATUS_ROOT/launcher.failed"; fi' EXIT
main
