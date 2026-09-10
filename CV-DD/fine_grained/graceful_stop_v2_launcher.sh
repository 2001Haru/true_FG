#!/usr/bin/env bash
set -euo pipefail

V2_ROOT="${V2_ROOT:-/linxi/dataset/FG_SRe2L_standard/v2}"
LAUNCHER_PID="${LAUNCHER_PID:?set LAUNCHER_PID}"
WORKER_PIDS="${WORKER_PIDS:?set WORKER_PIDS}"
WAIT_TASKS="${WAIT_TASKS:?set WAIT_TASKS to space-separated task ids}"
WORKER_PIDS="${WORKER_PIDS//,/ }"
WAIT_TASKS="${WAIT_TASKS//,/ }"
STOP_NAME="${STOP_NAME:-graceful_stop}"
POLL_SECONDS="${POLL_SECONDS:-30}"

for task in $WAIT_TASKS; do
    while [[ ! -f "$V2_ROOT/status/${task}.complete" ]]; do
        if [[ -f "$V2_ROOT/status/${task}.failed" ]]; then
            echo "task failed before graceful stop: $task" >&2
            exit 1
        fi
        sleep "$POLL_SECONDS"
    done
done

# All task children have written audited results and returned.  Stop the
# launcher before resuming TERM-pending workers so it cannot run an incomplete
# final summary or dispatch more work.
kill -TERM "$LAUNCHER_PID" 2>/dev/null || true
for worker in $WORKER_PIDS; do
    kill -TERM "$worker" 2>/dev/null || true
    kill -CONT "$worker" 2>/dev/null || true
done
sleep 2
rm -f "$V2_ROOT/status/launcher.running"
echo "$(date --iso-8601=seconds) stopped after tasks: $WAIT_TASKS" \
    > "$V2_ROOT/status/${STOP_NAME}.complete"
