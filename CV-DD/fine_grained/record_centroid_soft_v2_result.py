"""Audit Aircraft DINO-centroid Soft-v2 result provenance."""

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
    if not math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-12):
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--fkd", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--ipc", required=True, choices=(1, 3, 5), type=int)
    parser.add_argument("--student-seed", required=True, type=int)
    args = parser.parse_args()
    result = load(args.result)
    audit_payload(result, 100, 3333)
    expect(result["student_protocol_name"], "standard_protocol_v2", "protocol")
    expect(result["student_initialization"], "imagenet-v1", "initialization")
    expect(result["student_seed"], args.student_seed, "Student seed")
    expect(result["batch_size"], 20, "batch")
    expect(result["gradient_accumulation_steps"], 2, "accumulation")
    expect(result["epochs"], 400, "epochs")
    expect(result["total_optimizer_updates"], 400 * ((100 * args.ipc + 19) // 20), "updates")
    close(result["backbone_learning_rate"], 1e-4, "backbone LR")
    close(result["head_learning_rate"], 1e-3, "head LR")
    close(result["weight_decay"], 1e-5, "weight decay")
    expect(result["scheduler"], "cosine_annealing", "scheduler")
    expect(result["scheduler_t_max"], 400, "T_max")
    close(result["temperature"], 20, "temperature")
    expect(result["temperature_squared_multiplier"], False, "T squared")
    expect(result["validation_images"], 3333, "validation images")
    manifest = load(args.manifest)
    expect(manifest["status"], "complete", "selection")
    expect(manifest["method"], "dino_global_centroid_top_ipc", "method")
    expect(manifest["ipc"], args.ipc, "selection IPC")
    expect(manifest["selected_images"], 100 * args.ipc, "selected images")
    expect(manifest["ipc1_rank_identity_verified"], True, "rank1 identity")
    relabel = load(args.fkd / "relabel_manifest.json")
    expect(relabel["teacher_mode"], "train", "BSSL")
    expect(relabel["teacher_forward_split"], [10, 10] if args.ipc != 1 else [10, 10], "Teacher split")
    expect(relabel["epochs"], 400, "Relabel epochs")
    expect(relabel["batch_size"], 20, "Relabel batch")
    expect(relabel["teacher_sha256"], sha256(args.teacher), "Teacher hash")
    result["dino_centroid_soft_v2"] = {
        "status": "complete", "ipc": args.ipc, "selection_seed": None,
        "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256(args.manifest),
        "teacher": str(args.teacher.resolve()), "teacher_sha256": sha256(args.teacher),
    }
    temporary = args.result.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.result)


if __name__ == "__main__":
    main()
