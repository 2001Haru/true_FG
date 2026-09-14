#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_ViT_Teachers/aircraft_seed42_224_v1}"
SOURCE_ROOT="$EXP_ROOT/sources"
WEIGHT_DIR=/linxi/models/vit/imagenet21k
WEIGHTS="$WEIGHT_DIR/ViT-B_16.npz"
TRAIN_INDEX=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/train
RAW_IMAGES=/linxi/dataset/FD2/raw/fgvc-aircraft-2013b/data/images
TEST_ROOT=/linxi/dataset/FG_SRe2L_repro/v1/datasets/A_imsize224/test
R0_IMAGES=/linxi/dataset/FG_CoDA_standard/v2/baselines/random_real_standard/selected/A_imsize224/rseed0/ipc3
R0_FKD=/linxi/dataset/FG_HardAugFactorial_v2/aircraft_ipc3_v1/fkd/fullframe_cutmix/random_real/source_seed0/ipc3_bs20_ipc3
R0_RESULTS=/linxi/dataset/FG_SoftAugFactorial_v2/aircraft_ipc3_v1/results/fullframe_cutmix/random_real/source_seed0
PLAIN_COMMIT=460a162767de1722a014ed2261463dbbc01196b6
TRANSFG_COMMIT=9336fba46a4ac8ed2e33072c5f74ab459b114e4a

mkdir -p "$EXP_ROOT"/{logs,status,locks,preflight,teachers,fkd,results,post_eval,summary,audits} "$SOURCE_ROOT" "$WEIGHT_DIR"
ts(){ date --iso-8601=seconds; }

clone_pinned(){
  local url="$1" destination="$2" commit="$3"
  if [[ ! -d "$destination/.git" ]]; then
    git clone --filter=blob:none "$url" "$destination"
  fi
  if [[ "$(git -C "$destination" rev-parse HEAD)" != "$commit" ]]; then
    git -C "$destination" fetch origin "$commit" --depth=1
    git -C "$destination" checkout --detach "$commit"
  fi
  test "$(git -C "$destination" rev-parse HEAD)" = "$commit"
}

prepare_sources(){
  python -c 'import ml_collections'
  clone_pinned https://github.com/jeonsworld/ViT-pytorch.git "$SOURCE_ROOT/ViT-pytorch" "$PLAIN_COMMIT"
  clone_pinned https://github.com/TACJu/TransFG.git "$SOURCE_ROOT/TransFG" "$TRANSFG_COMMIT"
  if [[ ! -s "$WEIGHTS" ]]; then
    tmp="$WEIGHTS.download.$$"
    curl --fail --location --retry 5 --retry-delay 5 \
      https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz \
      --output "$tmp"
    mv "$tmp" "$WEIGHTS"
  fi
  sha256sum "$WEIGHTS" > "$EXP_ROOT/audits/ViT-B_16.npz.sha256"
  git -C "$SOURCE_ROOT/ViT-pytorch" rev-parse HEAD > "$EXP_ROOT/audits/vit_source_commit.txt"
  git -C "$SOURCE_ROOT/TransFG" rev-parse HEAD > "$EXP_ROOT/audits/transfg_source_commit.txt"
}

run_preflight(){
  local pids=() failed=0
  for spec in "vit:0:ViT-pytorch" "transfg:1:TransFG"; do
    IFS=: read -r kind gpu source <<< "$spec"
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -u "$ROOT_DIR/CV-DD/fine_grained/train_aircraft_vit_teacher.py" \
      --kind "$kind" --source-root "$SOURCE_ROOT/$source" --weights "$WEIGHTS" \
      --train-index-root "$TRAIN_INDEX" --raw-image-root "$RAW_IMAGES" --test-root "$TEST_ROOT" \
      --output-dir "$EXP_ROOT/preflight/$kind" --seed 42 --workers 8 --preflight-only \
      > "$EXP_ROOT/logs/preflight_${kind}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 ))
  python - "$EXP_ROOT" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])
for kind in ('vit','transfg'):
    x=json.load(open(root/'preflight'/kind/'training_manifest.json'))
    assert x['status']=='preflight_complete', x
    assert x['preflight']['batch_size']==64, x
    assert x['preflight']['max_cuda_memory_gib'] < 39.0, x
print('preflight gate passed')
PY
  echo "$(ts) complete" > "$EXP_ROOT/status/preflight.complete"
}

train_teachers(){
  local pids=() failed=0
  for spec in "vit:0:ViT-pytorch" "transfg:1:TransFG"; do
    IFS=: read -r kind gpu source <<< "$spec"
    if python - "$EXP_ROOT/teachers/$kind/training_manifest.json" <<'PY'
import json,sys
try: assert json.load(open(sys.argv[1]))['status']=='complete'
except Exception: raise SystemExit(1)
PY
    then continue; fi
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -u "$ROOT_DIR/CV-DD/fine_grained/train_aircraft_vit_teacher.py" \
      --kind "$kind" --source-root "$SOURCE_ROOT/$source" --weights "$WEIGHTS" \
      --train-index-root "$TRAIN_INDEX" --raw-image-root "$RAW_IMAGES" --test-root "$TEST_ROOT" \
      --output-dir "$EXP_ROOT/teachers/$kind" --seed 42 --workers 8 \
      > "$EXP_ROOT/logs/train_${kind}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 ))
  echo "$(ts) complete" > "$EXP_ROOT/status/teachers.complete"
}

rescore_fkd(){
  local pids=() failed=0
  for spec in "vit:0:ViT-pytorch" "transfg:1:TransFG"; do
    IFS=: read -r kind gpu source <<< "$spec"
    output="$EXP_ROOT/fkd/$kind/ipc3_bs20_ipc3"
    CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
      OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python -u "$ROOT_DIR/CV-DD/fine_grained/rescore_fkd_vit_teacher.py" \
      --kind "$kind" --source-root "$SOURCE_ROOT/$source" \
      --checkpoint "$EXP_ROOT/teachers/$kind/final_step10000.pth" \
      --image-root "$R0_IMAGES" --source-fkd "$R0_FKD" --output-fkd "$output" \
      --epochs 400 --batch-size 20 --workers 8 --seed 42 --fkd-seed 42 --skip-completed \
      > "$EXP_ROOT/logs/relabel_${kind}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 ))
  for kind in vit transfg; do
    output="$EXP_ROOT/fkd/$kind/ipc3_bs20_ipc3"
    python "$ROOT_DIR/CV-DD/fine_grained/audit_fkd.py" \
      --fkd-dir "$output" --images 300 --classes 100 --batch-size 20 --epochs 400 \
      --output "$EXP_ROOT/audits/${kind}_fkd_structure.json" >> "$EXP_ROOT/logs/relabel_${kind}.log" 2>&1
    python "$ROOT_DIR/CV-DD/fine_grained/audit_fkd_metadata_alignment.py" \
      --reference "$R0_FKD" --candidate "$output" \
      --output "$EXP_ROOT/audits/${kind}_fkd_vs_r0_metadata.json" >> "$EXP_ROOT/logs/relabel_${kind}.log" 2>&1
  done
  echo "$(ts) complete" > "$EXP_ROOT/status/fkd.complete"
}

run_students(){
  local running=() running_gpu=() failed=0 launched=0
  for kind in vit transfg; do
    for seed in 42 43 44; do
      gpu=$((launched % 2))
      result="$EXP_ROOT/results/$kind/ipc3_sseed${seed}.json"
      if [[ -s "$result" ]]; then continue; fi
      mkdir -p "$(dirname "$result")" "$EXP_ROOT/post_eval/$kind/sseed${seed}"
      CUDA_VISIBLE_DEVICES="$gpu" PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
        OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python -u "$ROOT_DIR/CV-DD/validate/train_fkd.py" \
        --model ResNet18 --ipc 3 --exp-name "R0_${kind}_teacher_s${seed}" \
        --original-data-path "$R0_IMAGES" --fkd-path "$EXP_ROOT/fkd/$kind/ipc3_bs20_ipc3" \
        --output-dir "$EXP_ROOT/post_eval/$kind/sseed${seed}" --batch-size 20 --epochs 400 \
        --dataset-name A_imsize224 --gradient-accumulation-steps 2 --mix-type cutmix \
        --workers 8 --persistent-workers --fkd_seed 42 --train-seed "$seed" --temperature 20 \
        --student-initialization imagenet-v1 --student-protocol-name standard_protocol_v2_vit_labeler \
        --adamw-weight-decay 1e-5 --adamw-beta1 .9 --adamw-beta2 .999 --adamw-eps 1e-8 \
        --adamw-backbone-lr 1e-4 --adamw-head-lr 1e-3 --cosine-t-max 400 --cosine-eta-min 0 \
        --val-dir "$TEST_ROOT" --disable-wandb --per-class-output "$result" \
        > "$EXP_ROOT/logs/eval_${kind}_s${seed}.log" 2>&1 &
      running+=("$!")
      running_gpu+=("$gpu")
      launched=$((launched + 1))
      if (( ${#running[@]} == 4 )); then
        for pid in "${running[@]}"; do wait "$pid" || failed=1; done
        running=(); running_gpu=()
      fi
    done
  done
  for pid in "${running[@]}"; do wait "$pid" || failed=1; done
  (( failed == 0 ))
  echo "$(ts) complete" > "$EXP_ROOT/status/students.complete"
}

main(){
  exec 9>"$EXP_ROOT/locks/launcher.lock"
  flock -n 9 || exit 75
  rm -f "$EXP_ROOT/status/launcher.failed" "$EXP_ROOT/status/launcher.complete"
  echo "$(ts) source_and_weight_preparation" > "$EXP_ROOT/status/launcher.running"
  prepare_sources > "$EXP_ROOT/logs/prepare.log" 2>&1
  echo "$(ts) preflight" > "$EXP_ROOT/status/launcher.running"
  run_preflight
  echo "$(ts) teacher_training" > "$EXP_ROOT/status/launcher.running"
  train_teachers
  echo "$(ts) fkd_rescore" > "$EXP_ROOT/status/launcher.running"
  rescore_fkd
  echo "$(ts) student_eval" > "$EXP_ROOT/status/launcher.running"
  run_students
  python "$ROOT_DIR/CV-DD/fine_grained/summarize_vit_teacher_labelers.py" \
    --root "$EXP_ROOT" --r0-result-root "$R0_RESULTS" > "$EXP_ROOT/logs/summary.log" 2>&1
  rm -f "$EXP_ROOT/status/launcher.running"
  echo "$(ts) complete" > "$EXP_ROOT/status/launcher.complete"
}

trap 'status=$?; if (( status != 0 )); then rm -f "$EXP_ROOT/status/launcher.running"; echo "$(ts) exit=$status" > "$EXP_ROOT/status/launcher.failed"; fi' EXIT
main
