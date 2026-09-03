#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FEATURE_SPACE="${1:?usage: coda_fg_queue.sh vae_space|dino_space}"
case "$FEATURE_SPACE" in
    vae_space) FEATURE_ARG=vae ;;
    dino_space) FEATURE_ARG=dinov2 ;;
    *) echo "Feature space must be vae_space or dino_space" >&2; exit 2 ;;
esac

BASE_ROOT="${CODA_BASE_ROOT:-/linxi/dataset/FG_CoDA_standard/v2}"
EXP_ROOT="$BASE_ROOT/$FEATURE_SPACE"
CACHE_ROOT="${CODA_CACHE_ROOT:-$BASE_ROOT/shared_feature_cache}"
DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
MODEL_ROOT="${CODA_MODEL_ROOT:-/linxi/models/CoDA/SDXL-Refiner}"
DINO_MODEL_ROOT="${CODA_DINO_MODEL_ROOT:-/linxi/models/DINOv2/dinov2-base}"
LOG_ROOT="$EXP_ROOT/logs"
STATUS_ROOT="$EXP_ROOT/status"
GLOBAL_LOCK_ROOT="$BASE_ROOT/locks"
DATASETS=(CUB_imsize224 A_imsize224 SC_imsize224)
IPCS=(1 3 5)
GENERATION_SEEDS=(0 1 2)
STUDENT_SEEDS=(42 43 44)
EVAL_WAVE_SIZE="${EVAL_WAVE_SIZE:-8}"
mkdir -p "$LOG_ROOT" "$STATUS_ROOT" "$GLOBAL_LOCK_ROOT" "$EXP_ROOT/summary"

timestamp() { date --iso-8601=seconds; }

write_definition() {
    local revision
    revision="$(git -C "$ROOT_DIR" rev-parse HEAD)"
    python - "$EXP_ROOT" "$revision" "$EVAL_WAVE_SIZE" "$FEATURE_SPACE" <<'PY'
import json, os, sys
from pathlib import Path
root=Path(sys.argv[1]); revision=sys.argv[2]; wave=int(sys.argv[3]); feature_space=sys.argv[4]
expected=[str((root/'results'/d/f'ipc{i}_gseed{g}_sseed{s}.json').resolve())
          for d in ('CUB_imsize224','A_imsize224','SC_imsize224')
          for i in (1,3,5) for g in (0,1,2) for s in (42,43,44)]
payload={
    'status':'not_started','method':'CoDA','feature_space':feature_space,
    'supervision':'hard_label_cross_entropy','git_revision':revision,
    'datasets':['CUB_imsize224','A_imsize224','SC_imsize224'],
    'ipcs':[1,3,5],'generation_seeds':[0,1,2],'student_seeds':[42,43,44],
    'discovery':{'n_neighbors':5,'min_cluster_size':2,'min_samples':1},
    'expected_generated_sets':27,'expected_results':81,
    'eval_wave_size':wave,'max_eval_concurrency_per_gpu':4,
    'sdxl_model_root':'/linxi/models/CoDA/SDXL-Refiner',
    'dino_model_root':'/linxi/models/DINOv2/dinov2-base' if feature_space=='dino_space' else None,
    'expected_result_files':expected,
}
path=root/'matrix_definition.json'; tmp=path.with_suffix('.json.tmp')
tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)
PY
}

run_stage() {
    local stage="$1" feature_space="$2" dataset="$3" ipc="$4" generation_seed="$5"
    mkdir -p "$LOG_ROOT/$dataset/ipc${ipc}/gseed${generation_seed}"
    CODA_BASE_ROOT="$BASE_ROOT" CODA_CACHE_ROOT="$CACHE_ROOT" DATA_ROOT="$DATA_ROOT" \
        CODA_MODEL_ROOT="$MODEL_ROOT" CODA_DINO_MODEL_ROOT="$DINO_MODEL_ROOT" \
        bash "$ROOT_DIR/CV-DD/fine_grained/run_coda_fg.sh" \
        "$stage" "$feature_space" "$dataset" "$ipc" "$generation_seed" \
        > "$LOG_ROOT/$dataset/ipc${ipc}/gseed${generation_seed}/${stage}_${feature_space}.log" 2>&1
}

run_eval() {
    local dataset="$1" ipc="$2" generation_seed="$3" student_seed="$4" gpu="$5"
    CUDA_VISIBLE_DEVICES="$gpu" CODA_BASE_ROOT="$BASE_ROOT" CODA_CACHE_ROOT="$CACHE_ROOT" \
        DATA_ROOT="$DATA_ROOT" CODA_MODEL_ROOT="$MODEL_ROOT" \
        CODA_DINO_MODEL_ROOT="$DINO_MODEL_ROOT" \
        bash "$ROOT_DIR/CV-DD/fine_grained/run_coda_fg.sh" \
        eval-hard "$FEATURE_SPACE" "$dataset" "$ipc" "$generation_seed" "$student_seed" \
        > "$LOG_ROOT/$dataset/ipc${ipc}/gseed${generation_seed}/eval_sseed${student_seed}.log" 2>&1
}

wait_wave() {
    local failed=0 pid
    for pid in "$@"; do wait "$pid" || failed=1; done
    (( failed == 0 ))
}

main() {
    # SDXL generation occupies both GPUs, so the two feature-space arms must be
    # serialized even though their outputs and per-result locks are isolated.
    exec 9>"$GLOBAL_LOCK_ROOT/launcher.lock"
    flock -n 9 || { echo "Another CoDA feature-space queue is already running" >&2; exit 1; }
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    export TORCH_HOME=/linxi/dataset/FD2/torch_cache
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
    for model_file in \
        "$MODEL_ROOT/sdxl-base/model_index.json" \
        "$MODEL_ROOT/sdxl-base/unet/diffusion_pytorch_model.fp16.safetensors" \
        "$MODEL_ROOT/sdxl-base/vae/diffusion_pytorch_model.fp16.safetensors" \
        "$MODEL_ROOT/sdxl-base/text_encoder/model.fp16.safetensors" \
        "$MODEL_ROOT/sdxl-base/text_encoder_2/model.fp16.safetensors"; do
        [[ -f "$model_file" ]] || { echo "missing SDXL component: $model_file" >&2; exit 1; }
    done
    if [[ "$FEATURE_ARG" == dinov2 ]]; then
        for model_file in "$DINO_MODEL_ROOT/model.safetensors" \
            "$DINO_MODEL_ROOT/config.json" "$DINO_MODEL_ROOT/preprocessor_config.json"; do
            [[ -f "$model_file" ]] || { echo "missing DINOv2 component: $model_file" >&2; exit 1; }
        done
    fi
    for dataset in "${DATASETS[@]}"; do
        [[ -d "$DATA_ROOT/$dataset/train" && -d "$DATA_ROOT/$dataset/test" ]] || {
            echo "missing prepared dataset: $DATA_ROOT/$dataset" >&2; exit 1;
        }
    done
    write_definition
    python - "$EXP_ROOT/matrix_definition.json" <<'PY'
import json, os, sys
from pathlib import Path
path=Path(sys.argv[1]); payload=json.loads(path.read_text(encoding='utf-8'))
payload['status']='running'; tmp=path.with_suffix('.json.tmp')
tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)
PY
    echo "$(timestamp) generation started" > "$STATUS_ROOT/generation.running"
    for dataset in "${DATASETS[@]}"; do
        if [[ "$FEATURE_ARG" == dinov2 ]]; then
            # DINO selects source images; path-aligned VAE latents still guide SDXL.
            run_stage features vae_space "$dataset" 1 0
        fi
        run_stage features "$FEATURE_SPACE" "$dataset" 1 0
        for ipc in "${IPCS[@]}"; do
            run_stage cluster "$FEATURE_SPACE" "$dataset" "$ipc" 0
            for generation_seed in "${GENERATION_SEEDS[@]}"; do
                run_stage generate "$FEATURE_SPACE" "$dataset" "$ipc" "$generation_seed"
                run_stage audit "$FEATURE_SPACE" "$dataset" "$ipc" "$generation_seed"
            done
        done
    done
    rm -f "$STATUS_ROOT/generation.running"
    echo "$(timestamp) generation complete" > "$STATUS_ROOT/generation.complete"

    echo "$(timestamp) evaluation started" > "$STATUS_ROOT/evaluation.running"
    local task_index=0 pids=() dataset ipc generation_seed student_seed gpu
    for dataset in "${DATASETS[@]}"; do
        for ipc in "${IPCS[@]}"; do
            for generation_seed in "${GENERATION_SEEDS[@]}"; do
                for student_seed in "${STUDENT_SEEDS[@]}"; do
                    gpu=$((task_index % 2))
                    run_eval "$dataset" "$ipc" "$generation_seed" "$student_seed" "$gpu" &
                    pids+=("$!")
                    task_index=$((task_index + 1))
                    if (( ${#pids[@]} == EVAL_WAVE_SIZE )); then
                        wait_wave "${pids[@]}"
                        pids=()
                    fi
                done
            done
        done
    done
    if (( ${#pids[@]} > 0 )); then wait_wave "${pids[@]}"; fi
    rm -f "$STATUS_ROOT/evaluation.running"
    echo "$(timestamp) evaluation complete" > "$STATUS_ROOT/evaluation.complete"
    python "$ROOT_DIR/CV-DD/fine_grained/summarize_coda_fg.py" \
        --experiment-root "$EXP_ROOT" --feature-space "$FEATURE_ARG" \
        --output "$EXP_ROOT/summary/coda_fg_hard_label.json" \
        > "$LOG_ROOT/summary.log" 2>&1
    python - "$EXP_ROOT/matrix_definition.json" <<'PY'
import json, os, sys
from pathlib import Path
path=Path(sys.argv[1]); payload=json.loads(path.read_text(encoding='utf-8'))
payload['status']='complete'; tmp=path.with_suffix('.json.tmp')
tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8'); os.replace(tmp,path)
PY
    echo "$(timestamp) launcher complete" > "$STATUS_ROOT/launcher.complete"
}

case "${2:-}" in
    --write-definition-only)
        write_definition
        ;;
    "")
        main
        ;;
    *)
        echo "usage: $0 vae_space|dino_space [--write-definition-only]" >&2
        exit 2
        ;;
esac
