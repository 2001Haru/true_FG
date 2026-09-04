#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODA_DIR="$ROOT_DIR/CoDA"
STAGE="${1:?usage: run_coda_fg.sh STAGE FEATURE_SPACE DATASET IPC GENERATION_SEED [STUDENT_SEED]}"
FEATURE_SPACE="${2:?missing feature space: vae_space or dino_space}"
DATASET="${3:?missing dataset}"
IPC="${4:?missing IPC}"
GENERATION_SEED="${5:-0}"
STUDENT_SEED="${6:-}"
GENERATION_GPU_COUNT="${CODA_GENERATION_GPU_COUNT:-$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)}"

case "$FEATURE_SPACE" in
    vae_space) FEATURE_ARG=vae ;;
    dino_space) FEATURE_ARG=dinov2 ;;
    *) echo "Feature space must be vae_space or dino_space" >&2; exit 2 ;;
esac

case "$DATASET" in
    CUB_imsize224|cub) DATASET=CUB_imsize224; SPEC=cub; CLASSES=200; VAL_IMAGES=5794; BATCH=20; ACCUM=2 ;;
    A_imsize224|aircraft|a) DATASET=A_imsize224; SPEC=aircraft; CLASSES=100; VAL_IMAGES=3333; BATCH=20; ACCUM=2 ;;
    SC_imsize224|cars|sc) DATASET=SC_imsize224; SPEC=cars; CLASSES=196; VAL_IMAGES=8041; BATCH=14; ACCUM=2 ;;
    *) echo "Unsupported dataset: $DATASET" >&2; exit 2 ;;
esac
[[ "$IPC" =~ ^(1|3|5)$ ]] || { echo "IPC must be 1, 3, or 5" >&2; exit 2; }
[[ "$GENERATION_SEED" =~ ^(0|1|2)$ ]] || { echo "generation seed must be 0, 1, or 2" >&2; exit 2; }

DATA_ROOT="${DATA_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1/datasets}"
MODEL_ROOT="${CODA_MODEL_ROOT:-/linxi/models/CoDA/SDXL-Refiner}"
DINO_MODEL_ROOT="${CODA_DINO_MODEL_ROOT:-/linxi/models/DINOv2/dinov2-base}"
BASE_ROOT="${CODA_BASE_ROOT:-/linxi/dataset/FG_CoDA_standard/v2}"
EXP_ROOT="$BASE_ROOT/$FEATURE_SPACE"
CACHE_ROOT="${CODA_CACHE_ROOT:-$BASE_ROOT/shared_feature_cache}"
DATA_DIR="$DATA_ROOT/$DATASET"
DISCOVERY_TAG="n5_mcs2_ms1"
ARM_ROOT="$EXP_ROOT/generation/$DATASET/$DISCOVERY_TAG/ipc${IPC}/gseed${GENERATION_SEED}"
CONFIG="$ARM_ROOT/generation_config.json"
CODA_OUTPUT="$EXP_ROOT/work/results/$SPEC/gseed${GENERATION_SEED}/Step-25/IPC-$IPC/DF-1.0-GTP-0.9-gamma-0.05/n_5_s_2"
SYNTHETIC_DIR="$CODA_OUTPUT/generated_images"
GENERATION_TRACE="$CODA_OUTPUT/generation_trace.json"
AUDIT="$ARM_ROOT/generation_audit.json"
GENERATED_PROVENANCE="$ARM_ROOT/generated_image_provenance.jsonl"
CLUSTER_AUDIT="$EXP_ROOT/clusters/$DATASET/$DISCOVERY_TAG/ipc${IPC}/cluster_audit.json"
RESULT_ROOT="${RESULT_ROOT:-$EXP_ROOT/results}"
POST_ROOT="${POST_EVAL_ROOT:-$EXP_ROOT/post_eval}"
LOG_ROOT="$EXP_ROOT/logs/$DATASET/ipc${IPC}/gseed${GENERATION_SEED}"
mkdir -p "$ARM_ROOT" "$LOG_ROOT"

fail() { echo "PRECHECK FAILED: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || fail "missing file: $1"; }
require_dir() { [[ -d "$1" ]] || fail "missing directory: $1"; }

if [[ ! -f "$CONFIG" ]]; then
    python "$ROOT_DIR/CV-DD/fine_grained/record_coda_generation_config.py" \
        --repo-root "$ROOT_DIR" --dataset-name "$DATASET" --spec "$SPEC" \
        --classes "$CLASSES" --ipc "$IPC" --data-dir "$DATA_DIR" \
        --generation-seed "$GENERATION_SEED" --generation-gpu-count "$GENERATION_GPU_COUNT" \
        --feature-space "$FEATURE_ARG" \
        --model-root "$MODEL_ROOT" --dino-model-root "$DINO_MODEL_ROOT" \
        --cache-root "$CACHE_ROOT" --output "$CONFIG" > "$LOG_ROOT/config.log"
fi

run_coda() {
    (cd "$CODA_DIR" && python -u CoDA_main.py \
        --program_path "$EXP_ROOT/work" --cache_root "$CACHE_ROOT" \
        --dataset_dir "$DATA_DIR" --local_model_path "$MODEL_ROOT" \
        --dino_model_path "$DINO_MODEL_ROOT" --feature_space "$FEATURE_ARG" \
        --spec "$SPEC" --nclass "$CLASSES" \
        --IPC "$IPC" --size 224 --seed "$GENERATION_SEED" \
        --generation_tag "gseed${GENERATION_SEED}" --n_neighbors 5 \
        --min_cluster_size 2 --min_samples 1 --num_seed_candidates 3 \
        --cluster_detial --cluster_logger --sample_step 25 \
        --denoising_factor 1.0 --guideTPercent 0.9 \
        --cfg_guidance_scale 5.0 --CoDA_guidance_scale 0.05 "$@")
}

chunks=$(((CLASSES + 9) / 10))
if [[ "$FEATURE_ARG" == vae ]]; then
    cluster_dir="$CACHE_ROOT/$SPEC"
else
    cluster_dir="$CACHE_ROOT/dinov2/$SPEC"
fi
feature_prefix="$cluster_dir/original_features_cache.pkl"
guidance_prefix="$CACHE_ROOT/$SPEC/original_features_cache.pkl"
FEATURE_AUDIT="$BASE_ROOT/cache_audits/$FEATURE_ARG/$DATASET.json"
GUIDANCE_AUDIT="$BASE_ROOT/cache_audits/vae/$DATASET.json"

case "$STAGE" in
    features)
        count=0
        [[ -d "$cluster_dir" ]] && count="$(find "$cluster_dir" -maxdepth 1 -type f -name 'original_features_cache.pkl_[0-9]*' | wc -l)"
        if [[ "$count" -ne "$chunks" ]]; then run_coda --calcu_features; fi
        count="$(find "$cluster_dir" -maxdepth 1 -type f -name 'original_features_cache.pkl_[0-9]*' | wc -l)"
        [[ "$count" -eq "$chunks" ]] || fail "feature chunks $count != $chunks"
        python "$ROOT_DIR/CV-DD/fine_grained/audit_coda_feature_cache.py" \
            --feature-space "$FEATURE_ARG" --cache-dir "$cluster_dir" \
            --data-dir "$DATA_DIR" --classes "$CLASSES" --output "$FEATURE_AUDIT"
        ;;
    cluster)
        require_file "$FEATURE_AUDIT"
        for ((i=0; i<chunks; i++)); do require_file "${feature_prefix}_${i}"; done
        if [[ "$FEATURE_ARG" == dinov2 ]]; then
            require_file "$GUIDANCE_AUDIT"
            for ((i=0; i<chunks; i++)); do require_file "${guidance_prefix}_${i}"; done
        fi
        count=0
        provenance_count=0
        [[ -d "$cluster_dir" ]] && count="$(find "$cluster_dir" -maxdepth 1 -type f -name "${IPC}_n_5_s_2_saved_clusters_[0-9]*.pkl" | wc -l)"
        [[ -d "$cluster_dir" ]] && provenance_count="$(find "$cluster_dir" -maxdepth 1 -type f -name "${IPC}_n_5_s_2_image_provenance_[0-9]*.jsonl" | wc -l)"
        if [[ "$count" -ne "$chunks" || "$provenance_count" -ne "$chunks" ]]; then
            run_coda --calcu_cluster
        fi
        count="$(find "$cluster_dir" -maxdepth 1 -type f -name "${IPC}_n_5_s_2_saved_clusters_[0-9]*.pkl" | wc -l)"
        provenance_count="$(find "$cluster_dir" -maxdepth 1 -type f -name "${IPC}_n_5_s_2_image_provenance_[0-9]*.jsonl" | wc -l)"
        [[ "$count" -eq "$chunks" ]] || fail "cluster chunks $count != $chunks"
        [[ "$provenance_count" -eq "$chunks" ]] || fail "provenance chunks $provenance_count != $chunks"
        python "$ROOT_DIR/CV-DD/fine_grained/audit_coda_clusters.py" \
            --cluster-dir "$cluster_dir" --data-dir "$DATA_DIR" \
            --classes "$CLASSES" --ipc "$IPC" \
            --feature-space "$FEATURE_ARG" --n-neighbors 5 --min-cluster-size 2 \
            --output "$CLUSTER_AUDIT"
        ;;
    generate)
        require_file "$CLUSTER_AUDIT"
        for ((i=0; i<chunks; i++)); do require_file "$cluster_dir/${IPC}_n_5_s_2_saved_clusters_${i}.pkl"; done
        count=0
        [[ -d "$SYNTHETIC_DIR" ]] && count="$(find "$SYNTHETIC_DIR" -type f -name '*.png' | wc -l)"
        if [[ "$count" -ne $((CLASSES * IPC)) || ! -f "$GENERATION_TRACE" ]]; then
            run_coda --generate_images
        fi
        count="$(find "$SYNTHETIC_DIR" -type f -name '*.png' | wc -l)"
        [[ "$count" -eq $((CLASSES * IPC)) ]] || fail "generated images $count != $((CLASSES * IPC))"
        require_file "$GENERATION_TRACE"
        ;;
    audit)
        require_dir "$SYNTHETIC_DIR"
        python "$ROOT_DIR/CV-DD/fine_grained/audit_coda_fg.py" \
            --dataset-name "$DATASET" --data-dir "$DATA_DIR" \
            --synthetic-dir "$SYNTHETIC_DIR" --ipc "$IPC" \
            --generation-config "$CONFIG" --cluster-audit "$CLUSTER_AUDIT" \
            --generation-trace "$GENERATION_TRACE" \
            --generated-provenance-output "$GENERATED_PROVENANCE" \
            --output "$AUDIT"
        ;;
    eval-hard)
        [[ "$STUDENT_SEED" =~ ^[0-9]+$ ]] || fail "eval-hard requires a numeric Student seed"
        require_file "$AUDIT"
        result="$RESULT_ROOT/$DATASET/ipc${IPC}_gseed${GENERATION_SEED}_sseed${STUDENT_SEED}.json"
        mkdir -p "$(dirname "$result")"
        command -v flock >/dev/null 2>&1 || fail "flock is required"
        exec 9>"${result}.lock"
        flock -n 9 || fail "evaluation already running: $result"
        if [[ ! -f "$result" ]]; then
            python -u "$ROOT_DIR/CV-DD/validate/train_fkd.py" \
                --hard-label --model ResNet18 --ipc "$IPC" \
                --exp-name "coda_${FEATURE_SPACE}_${DATASET}_ipc${IPC}_gseed${GENERATION_SEED}_sseed${STUDENT_SEED}" \
                --original-data-path "$SYNTHETIC_DIR" --output-dir "$POST_ROOT" \
                --batch-size "$BATCH" --epochs 400 --dataset-name "$DATASET" \
                --gradient-accumulation-steps "$ACCUM" --cos --workers 8 \
                --persistent-workers --fkd_seed 42 --train-seed "$STUDENT_SEED" \
                --student-initialization random --adamw-lr-override 1e-3 \
                --adamw-weight-decay 1e-5 --eta-override 2 --temperature 20 --min-scale 0.08 \
                --val-dir "$DATA_DIR/test" --disable-wandb --per-class-output "$result"
        fi
        python "$ROOT_DIR/CV-DD/fine_grained/annotate_coda_result.py" \
            --result "$result" --generation-audit "$AUDIT" \
            --generation-config "$CONFIG" --generation-seed "$GENERATION_SEED"
        python "$ROOT_DIR/CV-DD/fine_grained/audit_result.py" \
            --result "$result" --classes "$CLASSES" --validation-images "$VAL_IMAGES" \
            --expected-training-target hard_coarse_label
        ;;
    *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac
