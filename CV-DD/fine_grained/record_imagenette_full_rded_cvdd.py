"""Audit a full-RDED ImageNette result under current CV-DD semantics."""

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
    parser.add_argument("--ipc", required=True, type=int)
    parser.add_argument("--student-seed", required=True, type=int)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--fkd", required=True, type=Path)
    parser.add_argument("--generation-manifest", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    args = parser.parse_args()
    result = load(args.result)
    audit_payload(result, 10, 3925)
    expect(result["synthetic_data_path"], str(args.image_root.resolve()), "image root")
    expect(result["fkd_path"], str(args.fkd.resolve()), "FKD path")
    expect(result["student_initialization"], "random", "initialization")
    expect(result["student_seed"], args.student_seed, "seed")
    expect(result["optimizer"], "adamw", "optimizer")
    close(result["learning_rate"], 5e-4, "LR")
    close(result["weight_decay"], 0.01, "WD")
    expect(result["epochs"], 300, "epochs")
    expect(result["batch_size"], 10, "batch")
    expect(result["gradient_accumulation_steps"], 2, "accumulation")
    expect(result["total_optimizer_updates"], 300 * args.ipc, "updates")
    expect(result["mix_type"], "cutmix", "mix")
    close(result["temperature"], 20, "temperature")
    close(result["cosine_eta"], 1 if args.ipc == 10 else 2, "eta")
    expect(result["validation_images"], 3925, "validation count")
    generation = load(args.generation_manifest)
    expect(generation["status"], "complete", "generation")
    expect(generation["candidate_count_per_class"], 300, "candidates")
    expect(generation["num_crops_per_source"], 5, "crops")
    expect(generation["joint_topk"][str(args.ipc)], 4 * args.ipc, "joint top-k")
    expect(generation["source_pre_crop_resize"], "none; decode original resolution RGB", "pre-resize")
    expect(generation["teacher_mode"], "eval", "selection teacher")
    relabel = load(args.fkd / "relabel_manifest.json")
    expect(relabel["teacher_mode"], "train", "relabel teacher")
    expect(relabel["teacher_forward_split"], [5, 5], "relabel split")
    expect(relabel["epochs"], 300, "relabel epochs")
    expect(relabel["batch_size"], 10, "relabel batch")
    close(relabel["min_scale_crops"], 0.08, "relabel min scale")
    expect(relabel["teacher_sha256"], sha256(args.teacher), "teacher hash")
    result["full_rded_audit"] = {
        "status": "complete", "arm": "d_rded", "protocol": "current_cvdd",
        "generation_manifest": str(args.generation_manifest.resolve()),
        "generation_manifest_sha256": sha256(args.generation_manifest),
        "teacher_checkpoint": str(args.teacher.resolve()), "teacher_sha256": sha256(args.teacher),
    }
    temporary = args.result.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.result)


if __name__ == "__main__":
    main()
