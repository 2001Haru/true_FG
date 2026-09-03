#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODA_DIR="$ROOT_DIR/CoDA"
STAGE="${1:?usage: run_coda_fg.sh STAGE DATASET IPC GENERATION_SEED [STUDENT_SEED]}"
DATASET="${2:?missing dataset}"
IPC="${3:?missing IPC}"
GENERATION_SEED="${4:-0}"
STUDENT_SEED="${5:-}"

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
EXP_ROOT="${CODA_EXP_ROOT:-/linxi/dataset/FG_CoDA_standard/v1}"
DATA_DIR="$DATA_ROOT/$DATASET"
DISCOVERY_TAG="n15_mcs2_ms1"
ARM_ROOT="$EXP_ROOT/generation/$DATASET/$DISCOVERY_TAG/ipc${IPC}/gseed${GENERATION_SEED}"
CONFIG="$ARM_ROOT/generation_config.json"
CODA_OUTPUT="$EXP_ROOT/work/results/$SPEC/gseed${GENERATION_SEED}/Step-25/IPC-$IPC/DF-1.0-GTP-0.9-gamma-0.05/n_15_s_2"
SYNTHETIC_DIR="$CODA_OUTPUT/generated_images"
AUDIT="$ARM_ROOT/generation_audit.json"
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
        --generation-seed "$GENERATION_SEED" --model-root "$MODEL_ROOT" \
        --output "$CONFIG" > "$LOG_ROOT/config.log"
fi

run_coda() {
    (cd "$CODA_DIR" && python -u CoDA_main.py \
        --program_path "$EXP_ROOT/work" --dataset_dir "$DATA_DIR" \
        --local_model_path "$MODEL_ROOT" --spec "$SPEC" --nclass "$CLASSES" \
        --IPC "$IPC" --size 224 --seed "$GENERATION_SEED" \
        --generation_tag "gseed${GENERATION_SEED}" --n_neighbors 15 \
        --min_cluster_size 2 --min_samples 1 --num_seed_candidates 3 \
        --cluster_detial --cluster_logger --sample_step 25 \
        --denoising_factor 1.0 --guideTPercent 0.9 \
        --cfg_guidance_scale 5.0 --CoDA_guidance_scale 0.05 "$@")
}

chunks=$(((CLASSES + 9) / 10))
feature_prefix="$EXP_ROOT/work/results/clusterfile/$SPEC/original_features_cache.pkl"
cluster_dir="$EXP_ROOT/work/results/clusterfile/$SPEC"

case "$STAGE" in
    features)
        count=0
        [[ -d "$cluster_dir" ]] && count="$(find "$cluster_dir" -maxdepth 1 -type f -name 'original_features_cache.pkl_[0-9]*' | wc -l)"
        if [[ "$count" -ne "$chunks" ]]; then run_coda --calcu_features; fi
        count="$(find "$cluster_dir" -maxdepth 1 -type f -name 'original_features_cache.pkl_[0-9]*' | wc -l)"
        [[ "$count" -eq "$chunks" ]] || fail "feature chunks $count != $chunks"
        ;;
    cluster)
        for ((i=0; i<chunks; i++)); do require_file "${feature_prefix}_${i}"; done
        count=0
        [[ -d "$cluster_dir" ]] && count="$(find "$cluster_dir" -maxdepth 1 -type f -name "${IPC}_n_15_s_2_saved_clusters_[0-9]*.pkl" | wc -l)"
        if [[ "$count" -ne "$chunks" ]]; then run_coda --calcu_cluster; fi
        count="$(find "$cluster_dir" -maxdepth 1 -type f -name "${IPC}_n_15_s_2_saved_clusters_[0-9]*.pkl" | wc -l)"
        [[ "$count" -eq "$chunks" ]] || fail "cluster chunks $count != $chunks"
        ;;
    generate)
        for ((i=0; i<chunks; i++)); do require_file "$cluster_dir/${IPC}_n_15_s_2_saved_clusters_${i}.pkl"; done
        count=0
        [[ -d "$SYNTHETIC_DIR" ]] && count="$(find "$SYNTHETIC_DIR" -type f -name '*.png' | wc -l)"
        if [[ "$count" -ne $((CLASSES * IPC)) ]]; then run_coda --generate_images; fi
        count="$(find "$SYNTHETIC_DIR" -type f -name '*.png' | wc -l)"
        [[ "$count" -eq $((CLASSES * IPC)) ]] || fail "generated images $count != $((CLASSES * IPC))"
        ;;
    audit)
        require_dir "$SYNTHETIC_DIR"
        python "$ROOT_DIR/CV-DD/fine_grained/audit_coda_fg.py" \
            --dataset-name "$DATASET" --data-dir "$DATA_DIR" \
            --synthetic-dir "$SYNTHETIC_DIR" --ipc "$IPC" \
            --generation-config "$CONFIG" --output "$AUDIT"
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
                --exp-name "coda_${DATASET}_ipc${IPC}_gseed${GENERATION_SEED}_sseed${STUDENT_SEED}" \
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
            --result "$result" --classes "$CLASSES" --validation-images "$VAL_IMAGES"
        ;;
    *) echo "Unknown stage: $STAGE" >&2; exit 2 ;;
esac
