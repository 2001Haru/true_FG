"""Audit and annotate an Aircraft RDED single-quadrant Student-v2 result."""

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
    parser.add_argument("--generation-manifest", required=True, type=Path)
    parser.add_argument("--teacher-dir", required=True, type=Path)
    parser.add_argument("--fkd-dir", required=True, type=Path)
    parser.add_argument("--student-seed", required=True, type=int, choices=(42, 43, 44))
    args = parser.parse_args()

    result = load(args.result)
    audit_payload(result, 100, 3333)
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
    if any(abs(float(value)) > 1e-12 for value in result["final_scheduler_lrs"]):
        raise RuntimeError("final LR is nonzero")
    expect(result["epochs"], 400, "epochs")
    expect(result["batch_size"], 20, "batch")
    expect(result["gradient_accumulation_steps"], 2, "accumulation")
    expect_float(result["temperature"], 20, "temperature")
    expect(result["temperature_squared_multiplier"], False, "T squared")
    expect(result["hard_cross_entropy_mixture"], False, "hard CE")
    expect(result["mix_type"], "cutmix", "CutMix")

    generation = load(args.generation_manifest)
    expect(generation["status"], "complete", "generation status")
    expect(generation["generation_seed"], 42, "generation seed")
    expect(generation["method"], "released_rded_joint_topk_mosaic", "generation method")
    image_root = args.generation_manifest.parent / "ipc3"
    expect(result["synthetic_data_path"], str(image_root.resolve()), "image root")
    expect(result["fkd_path"], str(args.fkd_dir.resolve()), "FKD path")

    teacher_checkpoint = args.teacher_dir / "ResNet18.pth"
    teacher = load(args.teacher_dir / "complete.json")
    expect(teacher["status"], "complete", "Teacher status")
    expect(teacher["seed"], 42, "Teacher seed")
    expect(generation["teacher_sha256"], sha256(teacher_checkpoint), "generation Teacher")
    relabel = load(args.fkd_dir / "relabel_manifest.json")
    fkd = load(args.fkd_dir / "fkd_audit.json")
    quadrant = load(args.fkd_dir / "quadrant_audit.json")
    expect(relabel["status"], "complete", "Relabel status")
    expect(relabel["single_quadrant"], True, "single quadrant")
    expect(relabel["quadrant_seed"], 42, "quadrant seed")
    expect_float(relabel["min_scale_crops"], 0.5, "RRC minimum")
    expect_float(relabel["max_scale_crops"], 1.0, "RRC maximum")
    expect(relabel["epochs"], 400, "Relabel epochs")
    expect(relabel["batch_size"], 20, "Relabel batch")
    expect(relabel["teacher_mode"], "train", "Teacher mode")
    expect(relabel["teacher_sha256"], sha256(teacher_checkpoint), "Relabel Teacher")
    expect(fkd["status"], "complete", "FKD status")
    expect(fkd["images"], 300, "FKD images")
    expect(fkd["payload_entries"], 7, "FKD payload entries")
    expect(quadrant["status"], "complete", "quadrant audit")
    expect(quadrant["saved_schedule_mismatches"], 0, "quadrant replay mismatches")
    expect(quadrant["per_epoch_quadrant_counts_unique"], [[75, 75, 75, 75]], "epoch balance")
    expect(quadrant["per_image_quadrant_counts_unique"], [[100, 100, 100, 100]], "image balance")

    repository = Path(__file__).resolve().parents[2]
    result["rded_single_quadrant_v2"] = {
        "dataset": "A_imsize224",
        "ipc": 3,
        "teacher_seed": 42,
        "generation_seed": 42,
        "student_seed": args.student_seed,
        "source_images_per_epoch": 300,
        "quadrants_expanded": False,
        "single_quadrant_before_rrc": True,
        "quadrant_size": [112, 112],
        "rrc_output_size": [224, 224],
        "rrc_scale": [0.5, 1.0],
        "quadrant_seed": 42,
        "quadrant_schedule": relabel["quadrant_schedule"],
        "generation_manifest": str(args.generation_manifest.resolve()),
        "generation_manifest_sha256": sha256(args.generation_manifest),
        "relabel_manifest": str((args.fkd_dir / "relabel_manifest.json").resolve()),
        "relabel_manifest_sha256": sha256(args.fkd_dir / "relabel_manifest.json"),
        "quadrant_audit": str((args.fkd_dir / "quadrant_audit.json").resolve()),
        "quadrant_audit_sha256": sha256(args.fkd_dir / "quadrant_audit.json"),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
    }
    temporary = args.result.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.result)
    print(json.dumps({"status": "complete", "result": str(args.result.resolve())}))


if __name__ == "__main__":
    main()
