"""Audit and annotate an original-RDED-generation + Soft-v2 Aircraft result."""

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
    parser.add_argument("--ipc", required=True, type=int, choices=(1, 3, 5))
    parser.add_argument("--generation-seed", required=True, type=int, choices=(42, 43, 44))
    parser.add_argument("--student-seed", required=True, type=int, choices=(42, 43, 44))
    args = parser.parse_args()

    result = load(args.result)
    manifest = load(args.generation_manifest)
    audit_payload(result, 100, 3333)
    expect(manifest["status"], "complete", "generation status")
    expect(manifest["method"], "released_rded_joint_topk_mosaic", "method")
    expect(manifest["dataset"], "A_imsize224", "dataset")
    expect(manifest["generation_seed"], args.generation_seed, "generation seed")
    expect(manifest["factor"], 2, "factor")
    expect(manifest["candidate_count_per_class"], 66, "candidate count")
    expect(manifest["num_crops_per_source"], 5, "crop count")
    expect(len(manifest["outputs"][str(args.ipc)]), 100 * args.ipc, "generated image count")
    expect(manifest["pixel_optimization"], False, "pixel optimization")
    expect(manifest["bn_matching"], False, "BN matching")

    selected_root = args.generation_manifest.parent / f"ipc{args.ipc}"
    expect(result["synthetic_data_path"], str(selected_root.resolve()), "synthetic path")
    expect(result["fkd_path"], str(args.fkd_dir.resolve()), "FKD path")
    expect(result["student_protocol_name"], "standard_protocol_v2", "protocol")
    expect(result["student_initialization"], "imagenet-v1", "initialization")
    expect(result["student_seed"], args.student_seed, "Student seed")
    expect(result["training_target"], "fkd_soft_label", "training target")
    expect(result["optimizer"], "adamw", "optimizer")
    expect(result["optimizer_betas"], [0.9, 0.999], "betas")
    expect_float(result["optimizer_eps"], 1e-8, "eps")
    expect_float(result["backbone_learning_rate"], 1e-4, "backbone LR")
    expect_float(result["head_learning_rate"], 1e-3, "head LR")
    expect_float(result["weight_decay"], 1e-5, "matrix WD")
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
    expect(result["dataloader_workers"], 8, "workers")
    expect(result["persistent_workers"], True, "persistent workers")

    teacher = load(args.teacher_dir / "complete.json")
    teacher_checkpoint = args.teacher_dir / "ResNet18.pth"
    expect(teacher["status"], "complete", "Teacher status")
    expect(teacher["seed"], 42, "Teacher seed")
    expect(manifest["teacher_sha256"], sha256(teacher_checkpoint), "generation Teacher hash")
    relabel = load(args.fkd_dir / "relabel_manifest.json")
    fkd = load(args.fkd_dir / "fkd_audit.json")
    expect(relabel["status"], "complete", "Relabel status")
    expect(relabel["ipc"], args.ipc, "Relabel IPC")
    expect(relabel["teacher_mode"], "train", "Relabel Teacher mode")
    expect(relabel["epochs"], 400, "Relabel epochs")
    expect(relabel["batch_size"], 20, "Relabel batch")
    expect(relabel["workers"], 8, "Relabel workers")
    expect(relabel["persistent_workers"], False, "Relabel persistent")
    expect_float(relabel["temperature"], 20, "Relabel temperature")
    expect(relabel["mix_type"], "cutmix", "Relabel CutMix")
    expect(relabel["teacher_sha256"], sha256(teacher_checkpoint), "Relabel Teacher hash")
    expect(fkd["status"], "complete", "FKD status")
    expect(fkd["images"], 100 * args.ipc, "FKD images")

    protocol_path = Path(__file__).with_name("standard_v2_protocol.json")
    repository = Path(__file__).resolve().parents[2]
    result["standard_protocol"] = {
        "name": "fine_grained_sre2l_standard_protocol",
        "version": "v2",
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
        "dataset": "A_imsize224",
        "teacher_seed": 42,
        "generation_seed": args.generation_seed,
        "ipc": args.ipc,
        "student_seed": args.student_seed,
        "protocol_definition": str(protocol_path.resolve()),
        "protocol_definition_sha256": sha256(protocol_path),
    }
    result["rded_original_v2"] = {
        "method": manifest["method"],
        "generation_manifest": str(args.generation_manifest.resolve()),
        "generation_manifest_sha256": sha256(args.generation_manifest),
        "generation_seed": args.generation_seed,
        "candidate_count_per_class": 66,
        "num_crops_per_source": 5,
        "factor": 2,
        "joint_selected_crops": args.ipc * 4,
        "teacher_checkpoint": str(teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": sha256(teacher_checkpoint),
        "teacher_seed": 42,
        "relabel_manifest": str((args.fkd_dir / "relabel_manifest.json").resolve()),
        "relabel_manifest_sha256": sha256(args.fkd_dir / "relabel_manifest.json"),
        "fkd_audit": str((args.fkd_dir / "fkd_audit.json").resolve()),
        "fkd_audit_sha256": sha256(args.fkd_dir / "fkd_audit.json"),
    }
    temporary = args.result.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.result)
    print(json.dumps({"status": "complete", "result": str(args.result.resolve())}))


if __name__ == "__main__":
    main()
