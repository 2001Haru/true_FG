"""Audit B-image/A-label cross-view Aircraft IPC5 Student-v2 results."""

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from audit_result import audit_payload


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expect(actual, expected, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def expect_float(actual, expected: float, label: str, tolerance: float = 1e-12) -> None:
    if actual is None or not math.isclose(float(actual), expected, rel_tol=0, abs_tol=tolerance):
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--b-images", required=True, type=Path)
    parser.add_argument("--a-fkd", required=True, type=Path)
    parser.add_argument("--alignment-audit", required=True, type=Path)
    parser.add_argument("--construction-manifest", required=True, type=Path)
    parser.add_argument("--teacher-dir", required=True, type=Path)
    parser.add_argument("--student-seed", required=True, type=int, choices=(42, 43, 44))
    args = parser.parse_args()
    result = load(args.result)
    audit_payload(result, 100, 3333)
    expect(result["synthetic_data_path"], str(args.b_images.resolve()), "Student image path")
    expect(result["fkd_path"], str(args.a_fkd.resolve()), "Teacher-label FKD path")
    expect(result["student_protocol_name"], "standard_protocol_v2", "protocol")
    expect(result["student_initialization"], "imagenet-v1", "initialization")
    expect(result["student_seed"], args.student_seed, "Student seed")
    expect(result["training_target"], "fkd_soft_label", "training target")
    expect(result["optimizer"], "adamw", "optimizer")
    expect(result["optimizer_betas"], [0.9, 0.999], "betas")
    expect_float(result["optimizer_eps"], 1e-8, "eps")
    expect_float(result["backbone_learning_rate"], 1e-4, "backbone LR")
    expect_float(result["head_learning_rate"], 1e-3, "head LR")
    expect_float(result["weight_decay"], 1e-5, "weight decay")
    expect(result["scheduler"], "cosine_annealing", "scheduler")
    expect(result["scheduler_t_max"], 400, "T_max")
    expect_float(result["scheduler_eta_min"], 0, "eta_min")
    expect(result["epochs"], 400, "epochs")
    expect(result["batch_size"], 20, "batch")
    expect(result["gradient_accumulation_steps"], 2, "accumulation")
    expect_float(result["temperature"], 20, "temperature")
    expect(result["temperature_squared_multiplier"], False, "T squared")
    expect(result["hard_cross_entropy_mixture"], False, "hard CE")
    expect(result["mix_type"], "cutmix", "CutMix")

    alignment = load(args.alignment_audit)
    expect(alignment["status"], "complete", "alignment gate")
    expect(alignment["total_mismatches"], 0, "alignment mismatches")
    expect(alignment["compared_batches"], 10000, "aligned batches")
    expect(alignment["compared_examples"], 200000, "aligned examples")
    construction = load(args.construction_manifest)
    expect(construction["status"], "complete", "construction")
    relabel = load(args.a_fkd / "relabel_manifest.json")
    fkd = load(args.a_fkd / "fkd_audit.json")
    teacher_checkpoint = args.teacher_dir / "ResNet18.pth"
    expect(relabel["status"], "complete", "A Relabel")
    expect(relabel["teacher_mode"], "train", "A Teacher mode")
    expect_float(relabel["min_scale_crops"], 0.08, "RRC minimum")
    expect_float(relabel["max_scale_crops"], 1.0, "RRC maximum")
    expect(relabel["teacher_sha256"], sha256(teacher_checkpoint), "Teacher checkpoint")
    expect(fkd["status"], "complete", "A FKD")
    expect(fkd["images"], 500, "FKD images")
    expect(fkd["payload_entries"], 6, "FKD payload")

    repository = Path(__file__).resolve().parents[2]
    result["cross_view_label_v2"] = {
        "dataset": "A_imsize224", "ipc": 5, "student_seed": args.student_seed,
        "student_images": "B: H20 source images bilinear 224->112->224",
        "teacher_label_views": "A: original H20 224x224 images with identical geometry metadata",
        "intentional_teacher_student_clarity_mismatch": True,
        "rrc_flip_cutmix_metadata_exactly_aligned": True,
        "alignment_audit": str(args.alignment_audit.resolve()),
        "alignment_audit_sha256": sha256(args.alignment_audit),
        "construction_manifest": str(args.construction_manifest.resolve()),
        "construction_manifest_sha256": sha256(args.construction_manifest),
        "teacher_seed": 42, "rrc_scale": [0.08, 1.0],
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
    }
    temporary = args.result.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.result)
    print(json.dumps({"status": "complete", "result": str(args.result.resolve())}))


if __name__ == "__main__":
    main()
