"""Audit and annotate a Teacher-score-selected Aircraft Soft-v2 result."""

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
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=("entropy_t20", "true_ce_t1"))
    parser.add_argument("--ipc", required=True, type=int, choices=(1, 3, 5))
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
    expect(result["epochs"], 400, "epochs")
    expect(result["batch_size"], 20, "batch")
    expect(result["gradient_accumulation_steps"], 2, "accumulation")
    expect_float(result["temperature"], 20, "temperature")
    expect(result["temperature_squared_multiplier"], False, "T squared")
    expect(result["hard_cross_entropy_mixture"], False, "hard CE")
    expect(result["mix_type"], "cutmix", "CutMix")

    selection = load(args.selection_manifest)
    expect(selection["status"], "complete", "selection status")
    expect(selection["teacher_mode"], "eval", "selection Teacher mode")
    expect(selection["ranking"]["primary"], "entropy_t20 ascending", "primary ranking")
    image_root = args.selection_manifest.parent / "selected" / args.method / f"ipc{args.ipc}"
    expect(result["synthetic_data_path"], str(image_root.resolve()), "selected path")
    expect(result["fkd_path"], str(args.fkd_dir.resolve()), "FKD path")
    teacher_checkpoint = args.teacher_dir / "ResNet18.pth"
    expect(selection["teacher_sha256"], sha256(teacher_checkpoint), "selection Teacher")
    relabel = load(args.fkd_dir / "relabel_manifest.json")
    fkd = load(args.fkd_dir / "fkd_audit.json")
    entropy_audit = load(args.fkd_dir / "t20_entropy_audit.json")
    expect(relabel["status"], "complete", "Relabel status")
    expect(relabel["teacher_mode"], "train", "Relabel Teacher mode")
    expect(relabel["epochs"], 400, "Relabel epochs")
    expect(relabel["batch_size"], 20, "Relabel batch")
    expect_float(relabel["min_scale_crops"], 0.08, "RRC minimum")
    expect_float(relabel["max_scale_crops"], 1.0, "RRC maximum")
    expect(relabel["single_quadrant"], False, "single quadrant")
    expect(relabel["teacher_sha256"], sha256(teacher_checkpoint), "Relabel Teacher")
    expect(fkd["status"], "complete", "FKD status")
    expect(fkd["images"], 100 * args.ipc, "FKD images")
    expect(fkd["payload_entries"], 6, "FKD payload")
    expect(entropy_audit["status"], "complete", "entropy audit")
    expect_float(entropy_audit["temperature"], 20.0, "entropy temperature")

    repository = Path(__file__).resolve().parents[2]
    result["teacher_score_selection_v2"] = {
        "dataset": "A_imsize224",
        "ipc": args.ipc,
        "selection_method": args.method,
        "selection_teacher_mode": "eval",
        "selection_view": selection["input_view"],
        "teacher_seed": 42,
        "student_seed": args.student_seed,
        "selection_manifest": str(args.selection_manifest.resolve()),
        "selection_manifest_sha256": sha256(args.selection_manifest),
        "relabel_teacher_mode": "train",
        "rrc_scale": [0.08, 1.0],
        "fkd_t20_entropy_audit": str((args.fkd_dir / "t20_entropy_audit.json").resolve()),
        "fkd_t20_entropy_audit_sha256": sha256(args.fkd_dir / "t20_entropy_audit.json"),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
    }
    temporary = args.result.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.result)
    print(json.dumps({"status": "complete", "result": str(args.result.resolve())}))


if __name__ == "__main__":
    main()
