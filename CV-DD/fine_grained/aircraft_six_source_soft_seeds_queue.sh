#!/usr/bin/env bash
set -euo pipefail
ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EXP="${EXP_ROOT:-/tmp/fgdd_aircraft_six_source_soft_seed_completion}"
STAGE="${STAGE_ROOT:-/tmp/fgdd_aircraft_six_source_soft_seed_completion_stage}"
PARENT=/tmp/fgdd_aircraft_six_source_experiment
EXT=/tmp/fgdd_aircraft_six_source_extensions
TEACHER=/linxi/dataset/FG_SRe2L_standard/v1/teachers/A_imsize224/tseed42/ResNet18.pth
TEST=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/test
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$EXP"/{logs,status,locks,results,post_eval,summary,audits} "$STAGE"
exec 9>"$EXP/locks/launcher.lock";flock -n 9||exit 75
trap 's=$?;if((s));then rm -f "$EXP/status/running";echo "$(date --iso-8601=seconds) exit=$s">"$EXP/status/failed";fi' EXIT
rm -f "$EXP/status/failed" "$EXP/status/complete";date --iso-8601=seconds>"$EXP/status/running"
relabel(){
 local r=$1 g=$2 manifest out
 manifest="$EXT/construction/rseed$r/six_source_manifest.json";out="$STAGE/rseed$r"
 if [[ ! -f "$out/reference/relabel_manifest.json" ]];then
  CUDA_VISIBLE_DEVICES=$g python -u "$ROOT/CV-DD/fine_grained/relabel_aircraft_six_source.py" --manifest "$manifest" \
   --reference-fkd "$out/reference" --compressed-fkd "$out/compressed" --teacher "$TEACHER" \
   --epochs 400 --batch-size 20 --workers 8 --seed 42 > "$EXP/logs/relabel_rseed$r.log" 2>&1
 fi
 python "$ROOT/CV-DD/fine_grained/audit_aircraft_six_source_fkd.py" --manifest "$manifest" \
  --reference "$out/reference" --compressed "$out/compressed" --output "$EXP/audits/fkd_rseed$r.json" \
  > "$EXP/logs/audit_rseed$r.log" 2>&1
}
relabel 1 0 & p0=$!; relabel 2 1 & p1=$!; failed=0;wait $p0||failed=1;wait $p1||failed=1;((failed==0));date --iso-8601=seconds>"$EXP/status/relabel.complete"
soft(){
 local r=$1 mode=$2 seed=$3 gpu=$4 result manifest
 result="$EXP/results/rseed$r/$mode/sseed$seed.json";manifest="$EXT/construction/rseed$r/six_source_manifest.json"
 mkdir -p "$(dirname "$result")" "$EXP/post_eval/rseed$r/$mode/sseed$seed"
 CUDA_VISIBLE_DEVICES=$gpu python -u "$ROOT/CV-DD/validate/train_fkd.py" --model ResNet18 --ipc 3 \
  --exp-name "six_source_r${r}_${mode}_s$seed" --original-data-path "$EXT/construction/rseed$r/packed" \
  --fkd-path "$STAGE/rseed$r/$mode" --paired-source-manifest "$manifest" --paired-source-mode "$mode" \
  --output-dir "$EXP/post_eval/rseed$r/$mode/sseed$seed" --batch-size 20 --epochs 400 --dataset-name A_imsize224 \
  --gradient-accumulation-steps 2 --mix-type cutmix --workers 8 --persistent-workers --fkd_seed 42 \
  --train-seed "$seed" --temperature 20 --student-initialization imagenet-v1 \
  --student-protocol-name "standard_v2_six_source_fullframe_${mode}_r$r" --adamw-weight-decay 1e-5 \
  --adamw-beta1 .9 --adamw-beta2 .999 --adamw-eps 1e-8 --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 \
  --cosine-t-max 400 --cosine-eta-min 0 --val-dir "$TEST" --disable-wandb --per-class-output "$result" \
  > "$EXP/logs/soft_r${r}_${mode}_s$seed.log" 2>&1
}
pids=();index=0
for r in 1 2;do for mode in reference compressed;do for seed in 42 43 44;do
 soft $r $mode $seed $((index%2))&pids+=("$!");index=$((index+1))
 if((${#pids[@]}==4));then for pid in "${pids[@]}";do wait $pid||failed=1;done;pids=();((failed==0));fi
done;done;done
for pid in "${pids[@]}";do wait $pid||failed=1;done;((failed==0))
python "$ROOT/CV-DD/fine_grained/summarize_aircraft_six_source_soft_seeds.py" --root "$EXP" --parent "$PARENT">"$EXP/logs/summary.log" 2>&1
rm -f "$EXP/status/running";date --iso-8601=seconds>"$EXP/status/complete"
