"""Audit a standard-v2 Student result and attach immutable v1 upstream provenance."""

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

from audit_result import audit_payload


DATASETS = {
    "CUB_imsize224": (200, 5794, 20),
    "A_imsize224": (100, 3333, 20),
    "SC_imsize224": (196, 8041, 14),
}


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
    parser.add_argument("--reference-result", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=tuple(DATASETS))
    parser.add_argument("--teacher-seed", required=True, type=int)
    parser.add_argument("--recovery-seed", required=True, type=int)
    parser.add_argument("--ipc", required=True, type=int, choices=(1, 3, 5))
    parser.add_argument("--student-seed", required=True, type=int)
    args = parser.parse_args()

    classes, validation_images, batch = DATASETS[args.dataset]
    result = load(args.result)
    reference = load(args.reference_result)
    audit_payload(result, classes, validation_images)
    audit_payload(reference, classes, validation_images)

    reference_protocol = reference.get(
        "cub4k_protocol" if args.dataset == "CUB_imsize224" else "standard_protocol"
    )
    if not reference_protocol:
        raise RuntimeError("reference result lacks expected upstream provenance")
    expected_tuple = (args.teacher_seed, args.recovery_seed, args.ipc, args.student_seed)
    observed_tuple = tuple(reference_protocol[key] for key in (
        "teacher_seed", "recovery_seed", "ipc", "student_seed"
    ))
    expect(observed_tuple, expected_tuple, "reference tuple")
    if args.dataset == "CUB_imsize224":
        expect(reference_protocol["recovery_iterations"], 4000, "CUB recovery iterations")

    expect(result["student_protocol_name"], "standard_protocol_v2", "protocol name")
    expect(result["student_initialization"], "imagenet-v1", "Student initialization")
    expect(result["student_seed"], args.student_seed, "Student seed")
    expect(result["training_target"], "fkd_soft_label", "training target")
    expect(result["optimizer"], "adamw", "optimizer")
    expect(result["optimizer_betas"], [0.9, 0.999], "AdamW betas")
    expect_float(result["optimizer_eps"], 1e-8, "AdamW eps")
    expect_float(result["backbone_learning_rate"], 1e-4, "backbone LR")
    expect_float(result["head_learning_rate"], 1e-3, "head LR")
    expect_float(result["weight_decay"], 1e-5, "matrix weight decay")
    expect(result["epochs"], 400, "epochs")
    expect(result["scheduler"], "cosine_annealing", "scheduler")
    expect(result["scheduler_t_max"], 400, "scheduler T_max")
    expect_float(result["scheduler_eta_min"], 0.0, "scheduler eta_min")
    expect(result["scheduler_step_unit"], "epoch", "scheduler unit")
    expect(result["scheduler_step_timing"], "after_epoch", "scheduler timing")
    expect(result["cosine_eta"], None, "legacy cosine eta must be disabled")
    if any(abs(float(value)) > 1e-12 for value in result["final_scheduler_lrs"]):
        raise RuntimeError(f"final scheduler LR is not zero: {result['final_scheduler_lrs']}")
    expect(result["batch_size"], batch, "batch")
    expect(result["gradient_accumulation_steps"], 2, "accumulation")
    expect_float(result["temperature"], 20.0, "temperature")
    expect(result["temperature_squared_multiplier"], False, "T squared multiplier")
    expect(result["hard_cross_entropy_mixture"], False, "hard CE mixture")
    expect_float(result["label_smoothing"], 0.0, "label smoothing")
    expect(result["gradient_clipping"], None, "gradient clipping")
    expect(result["mix_type"], "cutmix", "CutMix replay")
    expect(result["fkd_seed"], 42, "FKD seed")
    expect(result["dataloader_workers"], 8, "workers")
    expect(result["persistent_workers"], True, "persistent workers")

    groups = {row["group_name"]: row for row in result["optimizer_parameter_groups"]}
    expect(set(groups), {"backbone_decay", "backbone_no_decay", "head_decay", "head_no_decay"}, "optimizer groups")
    for name, row in groups.items():
        expect_float(row["initial_lr"], 1e-3 if name.startswith("head_") else 1e-4, f"{name} LR")
        expect_float(row["weight_decay"], 1e-5 if name.endswith("_decay") and not name.endswith("no_decay") else 0.0, f"{name} WD")
    expect(groups["head_decay"]["parameter_names"], ["fc.weight"], "head decay parameters")
    expect(groups["head_no_decay"]["parameter_names"], ["fc.bias"], "head no-decay parameters")
    names = [name for row in groups.values() for name in row["parameter_names"]]
    if len(names) != len(set(names)):
        raise RuntimeError("optimizer parameter groups overlap")
    if any(name.endswith(".bias") or ".bn" in name for name in groups["backbone_decay"]["parameter_names"]):
        raise RuntimeError("bias/BN parameter entered a decay group")

    protocol_path = Path(__file__).resolve().with_name("standard_v2_protocol.json")
    repo_root = Path(__file__).resolve().parents[2]
    result["standard_protocol"] = {
        "name": "fine_grained_sre2l_standard_protocol",
        "version": "v2",
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip(),
        "dataset": args.dataset,
        "teacher_seed": args.teacher_seed,
        "recovery_seed": args.recovery_seed,
        "ipc": args.ipc,
        "student_seed": args.student_seed,
        "protocol_definition": str(protocol_path),
        "protocol_definition_sha256": sha256(protocol_path),
        "reference_v1_result": str(args.reference_result.resolve()),
        "reference_v1_result_sha256": sha256(args.reference_result),
        "upstream_provenance": reference_protocol,
    }
    temporary = args.result.with_suffix(args.result.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.result)
    print(json.dumps({"status": "complete", "result": str(args.result.resolve())}))


if __name__ == "__main__":
    main()
