"""Audit random-Student Aircraft H20 IPC5 A/B results using common A labels."""

import argparse
import json
import math
import os
from pathlib import Path

from audit_result import audit_payload


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def expect(actual, expected, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def expect_float(actual, expected: float, label: str, tolerance: float = 1e-12) -> None:
    if actual is None or not math.isclose(float(actual), expected, rel_tol=0, abs_tol=tolerance):
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--protocol", required=True, choices=("random_v2_lrs", "standard_v1"))
    parser.add_argument("--view", required=True, choices=("a_image_a_label", "b_image_a_label"))
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--a-fkd", required=True, type=Path)
    parser.add_argument("--alignment-audit", required=True, type=Path)
    parser.add_argument("--student-seed", required=True, type=int, choices=(42, 43, 44))
    args = parser.parse_args()
    result = load(args.result)
    audit_payload(result, 100, 3333)
    expect(result["synthetic_data_path"], str(args.image_root.resolve()), "image root")
    expect(result["fkd_path"], str(args.a_fkd.resolve()), "A FKD")
    expect(result["student_initialization"], "random", "initialization")
    expect(result["student_seed"], args.student_seed, "Student seed")
    expect(result["training_target"], "fkd_soft_label", "training target")
    expect(result["optimizer"], "adamw", "optimizer")
    expect(result["optimizer_betas"], [0.9, 0.999], "betas")
    expect_float(result["optimizer_eps"], 1e-8, "eps")
    expect_float(result["weight_decay"], 1e-5, "weight decay")
    expect(result["epochs"], 400, "epochs")
    expect(result["batch_size"], 20, "batch")
    expect(result["gradient_accumulation_steps"], 2, "accumulation")
    expect_float(result["temperature"], 20, "temperature")
    expect(result["temperature_squared_multiplier"], False, "T squared")
    expect(result["hard_cross_entropy_mixture"], False, "hard CE")
    expect(result["mix_type"], "cutmix", "CutMix")
    initial_hash = result.get("initial_model_sha256")
    if not isinstance(initial_hash, str) or len(initial_hash) != 64:
        raise RuntimeError("missing initialized-model SHA-256")
    if args.protocol == "random_v2_lrs":
        expect(result["student_protocol_name"], "standard_protocol_v2_random_init_only", "protocol name")
        expect_float(result["backbone_learning_rate"], 1e-4, "backbone LR")
        expect_float(result["head_learning_rate"], 1e-3, "head LR")
        expect(result["scheduler"], "cosine_annealing", "scheduler")
        expect(result["scheduler_t_max"], 400, "T_max")
        expect_float(result["scheduler_eta_min"], 0, "eta min")
        if any(abs(float(value)) > 1e-12 for value in result["final_scheduler_lrs"]):
            raise RuntimeError("v2 final LR is nonzero")
    else:
        expect(result["student_protocol_name"], "standard_protocol_v1", "protocol name")
        expect_float(result["learning_rate"], 1e-3, "uniform LR")
        expect(result["backbone_learning_rate"], None, "backbone LR")
        expect(result["head_learning_rate"], None, "head LR")
        expect(result["scheduler"], "cosine_lambda", "scheduler")
        expect_float(result["cosine_eta"], 2, "cosine eta")
        if any(not math.isclose(float(value), 5e-4, rel_tol=0, abs_tol=1e-10) for value in result["final_scheduler_lrs"]):
            raise RuntimeError(f"unexpected v1 final LR: {result['final_scheduler_lrs']}")
    alignment = load(args.alignment_audit)
    expect(alignment["status"], "complete", "alignment")
    expect(alignment["total_mismatches"], 0, "alignment mismatches")
    result["student_init_protocol_ablation"] = {
        "dataset": "A_imsize224", "ipc": 5, "protocol_arm": args.protocol,
        "view_arm": args.view, "student_seed": args.student_seed,
        "student_initialization": "torchvision ResNet18 weights=None including default BN",
        "common_teacher_labels": "A original H20 IPC5 FKD",
        "initial_model_sha256": initial_hash,
        "alignment_audit": str(args.alignment_audit.resolve()),
    }
    temporary = args.result.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.result)
    print(json.dumps({"status": "complete", "result": str(args.result.resolve()), "initial_model_sha256": initial_hash}))


if __name__ == "__main__":
    main()
