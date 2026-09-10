#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_TeacherScore_selection_v2/aircraft_v1}"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
TEACHER_DIR="${TEACHER_DIR:-/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42}"
RANDOM_REFERENCE_ROOT="${RANDOM_REFERENCE_ROOT:-/linxi/dataset/FG_RandomReal_soft_v2/aircraft_v1}"
GPUS=(${GPUS:-0 1})
STUDENT_SEEDS=(42 43 44)
IPCS=(1 3 5)
SELECTION_ROOT="$EXP_ROOT/selection"
SELECTION_MANIFEST="$SELECTION_ROOT/selection_manifest.json"
LOG_ROOT="$EXP_ROOT/logs"; STATUS_ROOT="$EXP_ROOT/status"; LOCK_ROOT="$EXP_ROOT/locks"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$LOCK_ROOT" "$EXP_ROOT/summary"

timestamp(){ date --iso-8601=seconds; }
wait_all(){ local failed=0 pid; for pid in "$@"; do wait "$pid" || failed=1; done; ((failed==0)); }

write_definition(){
  local ce_flag="$1"
  python - "$EXP_ROOT" "$DATA_ROOT" "$TEACHER_DIR" "$ce_flag" "$(git -C "$ROOT_DIR" rev-parse HEAD)" <<'PY'
import json,os,sys
from pathlib import Path
root,data,teacher=map(Path,sys.argv[1:4]); ce=bool(int(sys.argv[4])); revision=sys.argv[5]
tasks=[('entropy_t20',i,s) for i in (1,3,5) for s in (42,43,44)]
if ce: tasks += [('true_ce_t1',3,s) for s in (42,43,44)]
x={'status':'running','experiment':'aircraft_teacher_score_selection_soft_v2','git_revision':revision,
   'dataset':'A_imsize224','teacher_seed':42,'student_seeds':[42,43,44],
   'primary_selection':'per-class entropy_t20 ascending','primary_ipcs':[1,3,5],
   'diagnostic_scores':['entropy_t20','entropy_t1','true_ce_t1'],
   'conditional_ce_ipc3':ce,'rrc_scale':[0.08,1.0],
   'selection_teacher_mode':'eval','relabel_teacher_mode':'train',
   'data_root':str(data.resolve()),'teacher_dir':str(teacher.resolve()),
   'new_fkd_sets':3+int(ce),'new_student_trainings':len(tasks),
   'expected_result_files':[str((root/f'results/{m}/ipc{i}_sseed{s}.json').resolve()) for m,i,s in tasks],
   'summary':str((root/'summary/teacher_score_v2.json').resolve())}
p=root/'experiment_definition.json'; t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
}

relabel_one(){
  local method="$1" ipc="$2" gpu="$3"
  local images="$SELECTION_ROOT/selected/$method/ipc${ipc}"
  local base="$EXP_ROOT/fkd/$method/ipc${ipc}" actual="${base}_bs20_ipc${ipc}"
  local expected=$((400*100*ipc/20)) count=0 log="$LOG_ROOT/relabel_${method}_ipc${ipc}.log"
  [[ -d "$actual" ]] && count="$(find "$actual" -type f -name 'batch_*.tar' | wc -l)"
  if (( count != expected )); then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      python -u "$ROOT_DIR/CV-DD/relabel/relabel.py" --syn-data-path "$images" \
        --fkd-path "$base" --model-pool-dir "$TEACHER_DIR" --teacher-model-name ResNet18 \
        --gpu 0 --batch-size 20 --workers 8 --dataset-name A_imsize224 --epochs 400 \
        --seed 42 --fkd-seed 42 --min-scale-crops 0.08 --max-scale-crops 1 \
        --use-fp16 --mode fkd_save --mix-type cutmix > "$log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/audit_fkd.py" --fkd-dir "$actual" \
    --images $((100*ipc)) --classes 100 --batch-size 20 --epochs 400 >> "$log" 2>&1
  python "$ROOT_DIR/CV-DD/fine_grained/audit_fkd_t20_entropy.py" --fkd-dir "$actual" \
    --epochs 400 --temperature 20 --output "$actual/t20_entropy_audit.json" >> "$log" 2>&1
}

eval_one(){
  local method="$1" ipc="$2" student="$3" gpu="$4"
  local images="$SELECTION_ROOT/selected/$method/ipc${ipc}"
  local fkd="$EXP_ROOT/fkd/$method/ipc${ipc}_bs20_ipc${ipc}"
  local result="$EXP_ROOT/results/$method/ipc${ipc}_sseed${student}.json"
  local log="$LOG_ROOT/eval_${method}_ipc${ipc}_sseed${student}.log"
  mkdir -p "$(dirname "$result")"; exec 7>"${result}.lock"; flock -n 7 || return 75
  if [[ ! -f "$result" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -u "$ROOT_DIR/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc "$ipc" \
        --exp-name "teacher_score_${method}_A_ipc${ipc}_s${student}" \
        --original-data-path "$images" --fkd-path "$fkd" --output-dir "$EXP_ROOT/post_eval" \
        --batch-size 20 --epochs 400 --dataset-name A_imsize224 \
        --gradient-accumulation-steps 2 --mix-type cutmix --workers 8 --persistent-workers \
        --fkd_seed 42 --train-seed "$student" --temperature 20 \
        --student-initialization imagenet-v1 --student-protocol-name standard_protocol_v2 \
        --adamw-weight-decay 1e-5 --adamw-beta1 0.9 --adamw-beta2 0.999 --adamw-eps 1e-8 \
        --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 \
        --val-dir "$DATA_ROOT/A_imsize224/test" --disable-wandb --per-class-output "$result" \
        > "$log" 2>&1
  fi
  python "$ROOT_DIR/CV-DD/fine_grained/record_teacher_score_v2_result.py" \
    --result "$result" --selection-manifest "$SELECTION_MANIFEST" --method "$method" \
    --ipc "$ipc" --teacher-dir "$TEACHER_DIR" --fkd-dir "$fkd" --student-seed "$student" \
    >> "$log" 2>&1
}

main(){
  exec 9>"$LOCK_ROOT/launcher.lock"; flock -n 9 || exit 75
  rm -f "$STATUS_ROOT/launcher.failed" "$STATUS_ROOT/launcher.complete"
  echo "$(timestamp) started" > "$STATUS_ROOT/launcher.running"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -u "$ROOT_DIR/CV-DD/fine_grained/prepare_teacher_score_real_fg.py" \
      --data-dir "$DATA_ROOT/A_imsize224" --teacher "$TEACHER_DIR/ResNet18.pth" \
      --output-root "$SELECTION_ROOT" --batch-size 256 --workers 0 --skip-completed \
      > "$LOG_ROOT/selection.log" 2>&1
  local ce_flag
  ce_flag="$(python - "$SELECTION_MANIFEST" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); o=x['overlaps']['entropy_t20__vs__true_ce_t1']['3']
print(int(o['intersection'] < o['selected_per_method']))
PY
)"
  write_definition "$ce_flag"
  echo "$(timestamp) selection complete ce_ipc3=$ce_flag" > "$STATUS_ROOT/selection.complete"

  local tasks=("entropy_t20:1" "entropy_t20:3" "entropy_t20:5")
  (( ce_flag == 0 )) || tasks+=("true_ce_t1:3")
  local pids=() index=0 task method ipc gpu student
  for task in "${tasks[@]}"; do
    method="${task%%:*}"; ipc="${task##*:}"; gpu="${GPUS[$((index%${#GPUS[@]}))]}"
    relabel_one "$method" "$ipc" "$gpu" & pids+=("$!"); index=$((index+1))
    if (( ${#pids[@]} == ${#GPUS[@]} )); then wait_all "${pids[@]}"; pids=(); fi
  done
  if (( ${#pids[@]} )); then wait_all "${pids[@]}"; fi
  echo "$(timestamp) relabel and entropy audits complete" > "$STATUS_ROOT/relabel.complete"

  pids=(); index=0
  for task in "${tasks[@]}"; do
    method="${task%%:*}"; ipc="${task##*:}"
    for student in "${STUDENT_SEEDS[@]}"; do
      gpu="${GPUS[$((index%${#GPUS[@]}))]}"; eval_one "$method" "$ipc" "$student" "$gpu" & pids+=("$!"); index=$((index+1))
      if (( ${#pids[@]} == ${#GPUS[@]}*3 )); then wait_all "${pids[@]}"; pids=(); fi
    done
  done
  if (( ${#pids[@]} )); then wait_all "${pids[@]}"; fi
  python "$ROOT_DIR/CV-DD/fine_grained/summarize_teacher_score_v2.py" --root "$EXP_ROOT" \
    --random-reference-root "$RANDOM_REFERENCE_ROOT" > "$LOG_ROOT/summary.log" 2>&1
  python - "$EXP_ROOT/experiment_definition.json" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]); x=json.loads(p.read_text()); x['status']='complete'; t=p.with_suffix('.json.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n'); os.replace(t,p)
PY
  rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) complete" > "$STATUS_ROOT/launcher.complete"
}

trap 'rc=$?; if ((rc)); then rm -f "$STATUS_ROOT/launcher.running"; echo "$(timestamp) exit=$rc" > "$STATUS_ROOT/launcher.failed"; fi' EXIT
main
