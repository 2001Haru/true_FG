"""Audit ImageNette native-CVDD A/B resolution results."""

import argparse
import hashlib
import json
import math
import os
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
    parser.add_argument("--view", required=True, choices=("a_image", "b_image"))
    parser.add_argument("--ipc", required=True, type=int, choices=(10, 50))
    parser.add_argument("--student-seed", required=True, type=int, choices=(42, 43, 44))
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--a-fkd", required=True, type=Path)
    parser.add_argument("--construction-manifest", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--teacher-dir", required=True, type=Path)
    args = parser.parse_args()
    result = load(args.result)
    audit_payload(result, 10, 3925)
    expect(result["synthetic_data_path"], str(args.image_root.resolve()), "image root")
    expect(result["fkd_path"], str(args.a_fkd.resolve()), "A FKD")
    expect(result["student_protocol_name"], "imagenette_cvdd_native_resolution_v1", "protocol")
    expect(result["student_initialization"], "random", "initialization")
    expect(result["student_seed"], args.student_seed, "Student seed")
    expect(result["training_target"], "fkd_soft_label", "training target")
    expect(result["optimizer"], "adamw", "optimizer")
    expect_float(result["learning_rate"], 5e-4, "learning rate")
    expect_float(result["weight_decay"], 0.01, "weight decay")
    expect(result["optimizer_betas"], [0.9, 0.999], "betas")
    expect_float(result["optimizer_eps"], 1e-8, "eps")
    expect(result["epochs"], 300, "epochs")
    expect(result["batch_size"], 10, "batch")
    expect(result["gradient_accumulation_steps"], 2, "accumulation")
    expect(result["optimizer_updates_per_epoch"], args.ipc, "updates per epoch")
    expect(result["total_optimizer_updates"], 300 * args.ipc, "total updates")
    expect_float(result["temperature"], 20, "temperature")
    expect(result["temperature_squared_multiplier"], False, "T squared")
    expect(result["hard_cross_entropy_mixture"], False, "hard CE")
    expect(result["mix_type"], "cutmix", "CutMix")
    expect(result["scheduler"], "cosine_lambda", "scheduler")
    expected_eta = 1 if args.ipc == 10 else 2
    expect_float(result["cosine_eta"], expected_eta, "eta")
    expected_final_lr = 0.0 if args.ipc == 10 else 2.5e-4
    if any(not math.isclose(float(value), expected_final_lr, rel_tol=0, abs_tol=1e-10)
           for value in result["final_scheduler_lrs"]):
        raise RuntimeError(f"unexpected final LR: {result['final_scheduler_lrs']}")
    expect(result["final_epoch"], 299, "final epoch")
    if not isinstance(result.get("best_epoch"), int) or not 0 <= result["best_epoch"] <= 299:
        raise RuntimeError(f"invalid best epoch: {result.get('best_epoch')}")
    if not isinstance(result.get("initial_model_sha256"), str) or len(result["initial_model_sha256"]) != 64:
        raise RuntimeError("missing initial model hash")
    expect(result["validation_images"], 3925, "validation images")

    construction = load(args.construction_manifest)
    expect(construction["status"], "complete", "construction")
    expect(construction["selection"], "per-class entropy of softmax(raw_logits/20), ascending", "selection")
    expect_float(construction["selection_temperature"], 20, "selection temperature")
    expect(construction["selection_teacher_mode"], "eval", "selection Teacher mode")
    expect(construction["a_b_filename_and_imagefolder_index_aligned"], True, "A/B alignment")
    relabel = load(args.a_fkd / "relabel_manifest.json")
    fkd = load(args.a_fkd / "fkd_audit.json")
    expect(relabel["status"], "complete", "Relabel")
    expect(relabel["dataset_name"], "imagenet-nette", "Relabel dataset")
    expect(relabel["ipc"], args.ipc, "Relabel IPC")
    expect(relabel["teacher_mode"], "train", "BSSL")
    expect(relabel["epochs"], 300, "Relabel epochs")
    expect(relabel["batch_size"], 10, "Relabel batch")
    expect(relabel["teacher_forward_split"], [5, 5], "Teacher split")
    expect(relabel["workers"], 2, "Relabel workers")
    expect(relabel["persistent_workers"], False, "Relabel persistent")
    expect_float(relabel["min_scale_crops"], 0.08, "RRC minimum")
    expect_float(relabel["max_scale_crops"], 1.0, "RRC maximum")
    expect(relabel["mix_type"], "cutmix", "CutMix")
    expect(fkd["status"], "complete", "FKD")
    expect(fkd["images"], 10 * args.ipc, "FKD images")
    expect(fkd["batch_size"], 10, "FKD batch")
    expect(fkd["epochs"], 300, "FKD epochs")
    teacher_checkpoint = args.teacher_dir / "ResNet18.pth"
    expect(relabel["teacher_sha256"], sha256(teacher_checkpoint), "Teacher hash")
    protocol = load(args.protocol)
    expect(protocol["name"], "imagenette_cvdd_native_resolution_ablation", "protocol definition")
    result["imagenette_native_resolution"] = {
        "view": args.view, "ipc": args.ipc, "student_seed": args.student_seed,
        "selection": "per-class H20 entropy ascending", "common_a_labels": True,
        "batch_correction": "Validate 16 -> 10 to match paper and Relabel",
        "construction_manifest": str(args.construction_manifest.resolve()),
        "construction_manifest_sha256": sha256(args.construction_manifest),
        "protocol_definition": str(args.protocol.resolve()),
        "protocol_definition_sha256": sha256(args.protocol),
        "teacher_checkpoint": str(teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": sha256(teacher_checkpoint),
    }
    temporary = args.result.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.result)
    print(json.dumps({"status": "complete", "result": str(args.result.resolve()),
                      "best_epoch": result["best_epoch"], "initial_model_sha256": result["initial_model_sha256"]}))


if __name__ == "__main__":
    main()
