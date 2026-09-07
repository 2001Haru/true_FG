"""Create auditable unattended gates for the user-approved ImageWoof scan."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def audit_manifest(path, teacher_hash):
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = [0] * 10
    for row in payload["images"]:
        counts[int(row["class_id"])] += 1
    if payload.get("status") != "complete" or payload.get("selection_images") != 100 or counts != [10] * 10:
        raise RuntimeError(f"invalid manifest cardinality: {path}")
    if payload.get("teacher_checkpoint_sha256") != teacher_hash:
        raise RuntimeError(f"manifest Teacher mismatch: {path}")
    return sha256(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("base", "high", "top10"), required=True)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    args = parser.parse_args()
    root = args.experiment_root.resolve(); teacher = args.teacher.resolve()
    teacher_hash = sha256(teacher)
    if args.phase == "base":
        audit_path = root / "preflight/selection_preflight.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "complete_pending_manual_visual_review" or audit.get("teacher_checkpoint_sha256") != teacher_hash:
            raise RuntimeError("base preflight status/Teacher mismatch")
        manifests = {
            f"lambda_{value:+d}_rseed{seed}": audit_manifest(root / f"manifests/lambda_{value:+d}_rseed{seed}.json", teacher_hash)
            for value in (-4, -2, 0, 2, 4) for seed in (0, 1, 2)
        }
        gate_path = root / "preflight/manual_visual_gate.json"
        gate = {
            "approved": True, "reviewer": "user-approved-unattended-imagewoof-scan",
            "basis": "machine cardinality/provenance audit; contact sheets retained for retrospective review",
            "preflight_file": str(audit_path.resolve()), "preflight_sha256": sha256(audit_path),
            "teacher_checkpoint_sha256": teacher_hash, "manifest_sha256": manifests,
        }
    elif args.phase == "high":
        audit_path = root / "preflight/high_lambda_extension_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "complete_pending_high_lambda_review" or audit.get("teacher_checkpoint_sha256") != teacher_hash:
            raise RuntimeError("high-lambda preflight status/Teacher mismatch")
        manifests = {
            f"lambda_{value:+d}_rseed{seed}": audit_manifest(root / f"manifests/lambda_{value:+d}_rseed{seed}.json", teacher_hash)
            for value in (8, 16, 32) for seed in (0, 1, 2)
        }
        gate_path = root / "preflight/high_lambda_extension_gate.json"
        gate = {
            "approved": True, "reviewer": "user-approved-unattended-imagewoof-scan",
            "audit_file": str(audit_path.resolve()), "audit_sha256": sha256(audit_path),
            "teacher_checkpoint_sha256": teacher_hash, "manifest_sha256": manifests,
        }
    else:
        audit_path = root / "preflight/lambda0_entropy_allocation_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "complete_pending_allocation_review":
            raise RuntimeError("Top10 preflight status mismatch")
        manifest_path = root / "manifests/highest_entropy_top10_per_class.json"
        manifest_hash = audit_manifest(manifest_path, teacher_hash)
        gate_path = root / "preflight/lambda0_entropy_allocation_gate.json"
        gate = {
            "approved": True, "reviewer": "user-approved-unattended-imagewoof-scan",
            "audit_file": str(audit_path.resolve()), "audit_sha256": sha256(audit_path),
            "teacher_checkpoint_sha256": teacher_hash,
            "highest_entropy_top10_manifest_sha256": manifest_hash,
        }
    atomic(gate_path, gate)
    print(json.dumps({"status": "complete", "phase": args.phase, "gate": str(gate_path)}))


if __name__ == "__main__":
    main()
