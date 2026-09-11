"""Audit ImageNette RandomReal under the current CV-DD protocol."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from audit_result import audit_payload


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expect(actual, expected, label):
    if actual != expected:
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def close(actual, expected, label):
    if not math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-10):
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--ipc", required=True, choices=(10, 50), type=int)
    parser.add_argument("--student-seed", required=True, type=int)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--fkd", required=True, type=Path)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    args = parser.parse_args()
    result = load(args.result)
    audit_payload(result, 10, 3925)
    expect(result["synthetic_data_path"], str(args.image_root.resolve()), "image root")
    expect(result["fkd_path"], str(args.fkd.resolve()), "FKD")
    expect(result["student_seed"], args.student_seed, "seed")
    expect(result["student_initialization"], "random", "initialization")
    expect(result["optimizer"], "adamw", "optimizer")
    close(result["learning_rate"], 5e-4, "LR")
    close(result["weight_decay"], 0.01, "WD")
    expect(result["batch_size"], 10, "batch")
    expect(result["gradient_accumulation_steps"], 2, "accumulation")
    expect(result["total_optimizer_updates"], 300 * args.ipc, "updates")
    close(result["cosine_eta"], 1 if args.ipc == 10 else 2, "eta")
    close(result["temperature"], 20, "temperature")
    expect(result["validation_images"], 3925, "validation")
    manifest = load(args.selection_manifest)
    expect(manifest["status"], "complete", "selection")
    expect(manifest["method"], "random_real", "selection method")
    expect(manifest["ipc"], args.ipc, "selection IPC")
    expect(manifest["selection_seed"], 42, "selection seed")
    expect(manifest["selected_images"], 10 * args.ipc, "selected images")
    relabel = load(args.fkd / "relabel_manifest.json")
    expect(relabel["teacher_mode"], "train", "Teacher mode")
    expect(relabel["teacher_forward_split"], [5, 5], "Teacher split")
    expect(relabel["batch_size"], 10, "relabel batch")
    expect(relabel["epochs"], 300, "relabel epochs")
    expect(relabel["teacher_sha256"], sha256(args.teacher), "Teacher hash")
    result["random_real_audit"] = {
        "status": "complete", "selection_seed": 42, "protocol": "current_cvdd",
        "selection_manifest": str(args.selection_manifest.resolve()),
        "selection_manifest_sha256": sha256(args.selection_manifest),
    }
    temporary = args.result.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.result)


if __name__ == "__main__":
    main()
