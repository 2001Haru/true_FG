"""Audit and annotate an FD2 dual-semantics result evaluated by Student protocol v2."""

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from audit_result import audit_payload


def load(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expect(actual, expected, label):
    if actual != expected:
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def expect_float(actual, expected, label, tolerance=1e-12):
    if actual is None or not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--reference-v1-result", required=True, type=Path)
    parser.add_argument("--semantics", required=True, choices=("released_semantics", "paper_literal"))
    parser.add_argument("--ipc", required=True, type=int, choices=(1, 3, 5))
    parser.add_argument("--student-seed", required=True, type=int)
    args = parser.parse_args()
    result, reference = load(args.result), load(args.reference_v1_result)
    audit_payload(result, 100, 3333)
    audit_payload(reference, 100, 3333)
    expect(reference["fd2_semantics"], args.semantics, "reference semantics")
    expect(reference["student_seed"], args.student_seed, "reference Student seed")
    expect(reference["student_initialization"], "random", "reference Student initialization")
    expect(result["fd2_semantics"], args.semantics, "v2 semantics")
    expect(result["student_protocol_name"], "standard_protocol_v2", "protocol name")
    expect(result["student_initialization"], "imagenet-v1", "Student initialization")
    expect(result["student_seed"], args.student_seed, "Student seed")
    expect(result["training_target"], "fkd_soft_label", "training target")
    expect(result["synthetic_data_path"], reference["synthetic_data_path"], "synthetic image reuse")
    expect(result["fkd_path"], reference["fkd_path"], "FKD reuse")
    expect(result["fd2_teacher"], reference["fd2_teacher"], "FD2 Teacher reuse")
    expect(result["fd2_recovery_manifest"], reference["fd2_recovery_manifest"], "Recovery reuse")
    expect(result["optimizer"], "adamw", "optimizer")
    expect(result["optimizer_betas"], [0.9, 0.999], "AdamW betas")
    expect_float(result["optimizer_eps"], 1e-8, "AdamW eps")
    expect_float(result["backbone_learning_rate"], 1e-4, "backbone LR")
    expect_float(result["head_learning_rate"], 1e-3, "head LR")
    expect_float(result["weight_decay"], 1e-5, "matrix WD")
    expect(result["scheduler"], "cosine_annealing", "scheduler")
    expect(result["scheduler_t_max"], 400, "T_max")
    expect_float(result["scheduler_eta_min"], 0.0, "eta_min")
    expect(result["scheduler_step_unit"], "epoch", "scheduler unit")
    expect(result["scheduler_step_timing"], "after_epoch", "scheduler timing")
    if any(abs(float(value)) > 1e-12 for value in result["final_scheduler_lrs"]):
        raise RuntimeError("final scheduler LR is not zero")
    expect(result["epochs"], 400, "epochs")
    expect(result["batch_size"], 20, "batch")
    expect(result["gradient_accumulation_steps"], 2, "accumulation")
    expect_float(result["temperature"], 20.0, "temperature")
    expect(result["temperature_squared_multiplier"], False, "T squared")
    expect(result["hard_cross_entropy_mixture"], False, "hard CE mixture")
    expect(result["mix_type"], "cutmix", "CutMix")
    expect(result["dataloader_workers"], 8, "workers")
    expect(result["persistent_workers"], True, "persistent workers")
    groups = {row["group_name"]: row for row in result["optimizer_parameter_groups"]}
    expect(set(groups), {"backbone_decay", "backbone_no_decay", "head_decay", "head_no_decay"}, "groups")
    for name, row in groups.items():
        expect_float(row["initial_lr"], 1e-3 if name.startswith("head_") else 1e-4, f"{name} LR")
        decay = 0.0 if name.endswith("no_decay") else 1e-5
        expect_float(row["weight_decay"], decay, f"{name} WD")
    expect(groups["head_decay"]["parameter_names"], ["fc.weight"], "head weight")
    expect(groups["head_no_decay"]["parameter_names"], ["fc.bias"], "head bias")

    fkd_root = Path(result["fkd_path"])
    relabel_manifest = fkd_root / "relabel_manifest.json"
    fkd_audit = fkd_root / "fkd_audit.json"
    recovery_manifest = Path(result["fd2_recovery_manifest"])
    expect(load(relabel_manifest)["status"], "complete", "Relabel status")
    expect(load(fkd_audit)["status"], "complete", "FKD status")
    recovery = load(recovery_manifest)
    expect(recovery["status"], "complete", "Recovery status")
    expect(recovery["semantics"], args.semantics, "Recovery semantics")
    expect(recovery["iterations"], 4000, "Recovery iterations")

    protocol_path = Path(__file__).with_name("standard_v2_protocol.json")
    repo_root = Path(__file__).resolve().parents[2]
    result["standard_protocol"] = {
        "name": "fine_grained_sre2l_standard_protocol", "version": "v2",
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip(),
        "dataset": "A_imsize224", "teacher_seed": 42, "recovery_seed": 42,
        "ipc": args.ipc, "student_seed": args.student_seed,
        "protocol_definition": str(protocol_path.resolve()),
        "protocol_definition_sha256": sha256(protocol_path),
    }
    result["fd2_v2_reuse"] = {
        "reference_v1_result": str(args.reference_v1_result.resolve()),
        "reference_v1_result_sha256": sha256(args.reference_v1_result),
        "synthetic_data_path": result["synthetic_data_path"],
        "fkd_path": result["fkd_path"],
        "recovery_manifest": str(recovery_manifest.resolve()),
        "recovery_manifest_sha256": sha256(recovery_manifest),
        "relabel_manifest": str(relabel_manifest.resolve()),
        "relabel_manifest_sha256": sha256(relabel_manifest),
        "fkd_audit": str(fkd_audit.resolve()), "fkd_audit_sha256": sha256(fkd_audit),
    }
    temporary = args.result.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.result)
    print(json.dumps({"status": "complete", "result": str(args.result.resolve())}))


if __name__ == "__main__":
    main()
