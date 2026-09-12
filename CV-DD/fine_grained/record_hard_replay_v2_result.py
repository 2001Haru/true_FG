"""Audit a standard-v2 hard-label result that exactly replays a soft FKD trajectory."""

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
    parser.add_argument("--source-soft-result", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--source-seed", default="none")
    parser.add_argument("--ipc", required=True, choices=(1, 3, 5), type=int)
    parser.add_argument("--student-seed", required=True, type=int)
    parser.add_argument("--classes", default=100, type=int)
    parser.add_argument("--validation-images", default=3333, type=int)
    parser.add_argument("--expected-mix-type", default="cutmix", choices=("cutmix", "none"))
    parser.add_argument("--require-full-image-resize", action="store_true")
    args = parser.parse_args()
    result, source = load(args.result), load(args.source_soft_result)
    audit_payload(result, args.classes, args.validation_images,
                  expected_training_target="fkd_replay_hard_cutmix_ce")
    expect(result["student_protocol_name"], "standard_protocol_v2_hard_replay", "protocol")
    expect(result["student_initialization"], "imagenet-v1", "initialization")
    expect(result["student_seed"], args.student_seed, "Student seed")
    expect(result["training_target"], "fkd_replay_hard_cutmix_ce", "training target")
    expect(result["fkd_hard_label"], True, "hard replay flag")
    close(result["hard_label_student_temperature"], 1.0, "hard temperature")
    expected_mix = None if args.expected_mix_type == "none" else "cutmix"
    expect(result["mix_type"], expected_mix, "mix type")
    expect(result["hard_cutmix_lambda_source"],
           "actual_bbox_area" if expected_mix == "cutmix" else None, "CutMix lambda")
    expect(result["synthetic_data_path"], source["synthetic_data_path"], "image root")
    expect(result["fkd_path"], source["fkd_path"], "FKD trajectory")
    expect(result["batch_size"], 20, "batch")
    expect(result["gradient_accumulation_steps"], 2, "accumulation")
    expect(result["epochs"], 400, "epochs")
    close(result["backbone_learning_rate"], 1e-4, "backbone LR")
    close(result["head_learning_rate"], 1e-3, "head LR")
    close(result["weight_decay"], 1e-5, "weight decay")
    expect(result["scheduler"], "cosine_annealing", "scheduler")
    expect(result["scheduler_t_max"], 400, "T_max")
    expect(result["validation_images"], args.validation_images, "validation images")
    if not isinstance(result.get("initial_model_sha256"), str) or len(result["initial_model_sha256"]) != 64:
        raise RuntimeError("missing hard-run initial model SHA-256")
    if source.get("initial_model_sha256") is not None:
        expect(result["initial_model_sha256"], source["initial_model_sha256"], "paired initialization")
    fkd_audit = load(Path(result["fkd_path"]) / "fkd_audit.json")
    expect(fkd_audit["status"], "complete", "FKD audit")
    expect(fkd_audit["images"], args.classes * args.ipc, "FKD images")
    expect(fkd_audit["epochs"], 400, "FKD epochs")
    relabel = load(Path(result["fkd_path"]) / "relabel_manifest.json")
    expect(relabel["mix_type"], expected_mix, "Relabel mix type")
    if args.require_full_image_resize:
        expect(relabel.get("full_image_resize"), True, "full-image resize")
    result["hard_replay_v2"] = {
        "status": "complete", "method": args.method,
        "source_seed": None if args.source_seed == "none" else int(args.source_seed),
        "ipc": args.ipc, "student_seed": args.student_seed,
        "source_soft_result": str(args.source_soft_result.resolve()),
        "source_soft_result_sha256": sha256(args.source_soft_result),
        "source_soft_initial_hash_available": source.get("initial_model_sha256") is not None,
        "only_supervision_changed": True,
        "augmentation_and_cutmix_trajectory_source": result["fkd_path"],
    }
    temporary = args.result.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.result)


if __name__ == "__main__":
    main()
