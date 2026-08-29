#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:?usage: post_eval_queue.sh DATASET RECOVERY_SEED...}"
shift
[[ "$#" -gt 0 ]] || { echo "At least one recovery seed is required" >&2; exit 2; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_ROOT="${EXP_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1}"
WAIT_SECONDS="${WAIT_SECONDS:-300}"
LOG_ROOT="$EXP_ROOT/logs/jobs"
STATUS_ROOT="$EXP_ROOT/queue_status"
STUDENT_SEEDS=(42 43 44)
RECOVERY_SEEDS=("$@")
mkdir -p "$LOG_ROOT" "$STATUS_ROOT"

wait_for_recovery() {
    local recovery_seed="$1"
    local status="$STATUS_ROOT/${DATASET}_rseed${recovery_seed}.status"
    while [[ ! -f "$status" ]] || ! grep -q 'sampling complete' "$status"; do
        echo "$(date --iso-8601=seconds) waiting for completed recovery: $status" >&2
        sleep "$WAIT_SECONDS"
    done
}

for recovery_seed in "${RECOVERY_SEEDS[@]}"; do
    wait_for_recovery "$recovery_seed"
done

for recovery_seed in "${RECOVERY_SEEDS[@]}"; do
    for ipc in 1 3 5; do
        echo "$(date --iso-8601=seconds) relabel $DATASET rseed=$recovery_seed ipc=$ipc"
        bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" relabel "$DATASET" "$recovery_seed" "$ipc" \
            > "$LOG_ROOT/relabel_${DATASET}_rseed${recovery_seed}_ipc${ipc}.log" 2>&1
        for student_seed in "${STUDENT_SEEDS[@]}"; do
            echo "$(date --iso-8601=seconds) eval $DATASET rseed=$recovery_seed ipc=$ipc sseed=$student_seed"
            bash "$ROOT_DIR/fine_grained/run_sre2l_fg.sh" eval \
                "$DATASET" "$recovery_seed" "$ipc" "$student_seed" \
                > "$LOG_ROOT/eval_${DATASET}_rseed${recovery_seed}_ipc${ipc}_sseed${student_seed}.log" 2>&1
        done
    done
done

complete="$STATUS_ROOT/post_${DATASET}_rseeds_$(IFS=_; echo "${RECOVERY_SEEDS[*]}").complete"
echo "$(date --iso-8601=seconds) post-evaluation queue complete" > "$complete"
echo "$(date --iso-8601=seconds) post-evaluation queue complete: $DATASET seeds ${RECOVERY_SEEDS[*]}"
