#!/usr/bin/env bash
set -euo pipefail
while ! timeout 5 head -c 1 /linxi/true_FG/CV-DD/relabel/relabel.py >/dev/null 2>&1; do
  sleep 30
done
REPO_ROOT=/linxi/true_FG exec bash /tmp/aircraft_retarget_light_rrc_queue.sh
