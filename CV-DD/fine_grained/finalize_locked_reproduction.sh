#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_EXP_ROOT="${BASE_EXP_ROOT:-/linxi/dataset/FG_SRe2L_repro/v1}"
RESULT_ROOT="${RESULT_ROOT:-$BASE_EXP_ROOT/diagnostics/student_imagenet/results}"
LOG_ROOT="${LOG_ROOT:-$BASE_EXP_ROOT/diagnostics/student_imagenet/logs/jobs}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_EXP_ROOT/diagnostics/student_imagenet/summary}"
mkdir -p "$OUTPUT_DIR"

python "$ROOT_DIR/fine_grained/audit_protocol_provenance.py" \
    --base-root "$BASE_EXP_ROOT" --recovery-seeds 41 42 43 \
    --output "$OUTPUT_DIR/protocol_provenance.json"
python "$ROOT_DIR/fine_grained/summarize_locked_protocol.py" \
    --result-root "$RESULT_ROOT" --log-root "$LOG_ROOT" \
    --output-dir "$OUTPUT_DIR"
python "$ROOT_DIR/fine_grained/summarize_seed_variance.py" \
    --result-root "$RESULT_ROOT" --log-root "$LOG_ROOT" \
    --output-dir "$OUTPUT_DIR"
python "$ROOT_DIR/fine_grained/summarize_protocol_diagnostics.py" \
    --base-root "$BASE_EXP_ROOT" --output-dir "$OUTPUT_DIR"
python "$ROOT_DIR/fine_grained/audit_fd2_release_inventory.py" \
    --repo-root "$(dirname "$ROOT_DIR")" \
    --output "$OUTPUT_DIR/fd2_release_inventory.json"
python "$ROOT_DIR/fine_grained/build_reproduction_report.py" \
    --summary-dir "$OUTPUT_DIR" --output "$OUTPUT_DIR/reproduction_report.md"

git_revision="$(git -C "$ROOT_DIR" rev-parse HEAD)"
python - "$OUTPUT_DIR" "$git_revision" "$ROOT_DIR" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

output_dir = Path(sys.argv[1])
git_revision = sys.argv[2]
root_dir = Path(sys.argv[3])
inputs = {
    "protocol_provenance": output_dir / "protocol_provenance.json",
    "locked_results": output_dir / "locked_results.json",
    "seed_variance_results": output_dir / "seed_variance_results.json",
    "protocol_diagnostics": output_dir / "protocol_diagnostics.json",
    "fd2_release_inventory": output_dir / "fd2_release_inventory.json",
}
payloads = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in inputs.items()}
statuses = {name: payload["status"] for name, payload in payloads.items()}
artifacts = {}
for name, path in inputs.items():
    artifacts[name] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
report_path = output_dir / "reproduction_report.md"
artifacts["reproduction_report"] = {
    "path": str(report_path.resolve()),
    "bytes": report_path.stat().st_size,
    "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
}
frozen_audit_path = root_dir / "fine_grained" / "official_fd2_release_audit.json"
artifacts["historical_release_audit"] = {
    "path": str(frozen_audit_path.resolve()),
    "bytes": frozen_audit_path.stat().st_size,
    "sha256": hashlib.sha256(frozen_audit_path.read_bytes()).hexdigest(),
}
if any(status != "complete" for status in statuses.values()):
    raise RuntimeError(f"locked reproduction is incomplete: {statuses}")

completion = {
    "status": "complete",
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "git_revision": git_revision,
    "locked_student_matrix_runs": payloads["locked_results"]["completed_runs"],
    "crossed_seed_design_runs": payloads["seed_variance_results"]["completed_runs"],
    "requested_recovery_seeds": payloads["protocol_provenance"]["requested_recovery_seeds"],
    "artifacts": artifacts,
}
path = output_dir / "completion_audit.json"
path.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"status": "complete", "output": str(path.resolve())}, sort_keys=True))
PY
