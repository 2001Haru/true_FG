"""Strict audit for a non-target probability permutation result."""

import argparse
import json
import math
from pathlib import Path

from imagenette_entropy_protocol import CLASSES, file_sha256, lr_for_epoch


EVAL_EPOCHS = (200, 400, 600, 800, 1000, 1200, 1333, 1400, 1600, 1666, 1800, 2000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--selection-seed", choices=(0, 1, 2), required=True, type=int)
    parser.add_argument("--student-seed", choices=(42, 43, 44), required=True, type=int)
    parser.add_argument("--group", choices=("high", "low"), required=True)
    parser.add_argument("--experiment-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    supervision = f"{args.group}_entropy_permuted_soft1"
    original_supervision = f"{args.group}_entropy_soft1"
    spec = Path(__file__).resolve().with_name("imagenette_nontarget_permutation_protocol.json")
    manifest = root / "manifests" / f"lambda_{0:+d}_rseed{args.selection_seed}.json"
    permutation_file = root / "preflight" / "lambda0_nontarget_permutations.json"
    original_result = (
        root
        / "results"
        / "lambda0_entropy_allocation"
        / f"rseed{args.selection_seed}"
        / f"{original_supervision}_sseed{args.student_seed}.json"
    )
    original = json.loads(original_result.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "protocol": "imagenette_nontarget_permutation_v1",
        "protocol_spec_sha256": file_sha256(spec),
        "dataset": "imagenet-nette",
        "classes": CLASSES,
        "ipc": 10,
        "train_images": 100,
        "test_images": 3925,
        "selection_seed": args.selection_seed,
        "student_seed": args.student_seed,
        "supervision": supervision,
        "student_model": "CoDA_released_ResNetAP10",
        "student_initialization": "random",
        "batch_size": 64,
        "actual_batch_sizes_per_epoch": [64, 36],
        "loss_batch_denominator": "current_actual_batch_size",
        "gradient_accumulation_steps": 1,
        "drop_last": False,
        "epochs": 2000,
        "updates_completed": 4000,
        "cutmix": False,
        "mixup": False,
        "evaluation_epochs": list(EVAL_EPOCHS),
        "evaluation_count": 12,
        "teacher_and_student_view_pixels_identical": True,
        "teacher_mode": "eval_frozen_no_grad",
        "teacher_temperature": 1.0,
        "student_temperature": 1.0,
        "temperature_squared_multiplier": False,
        "loss": (
            "batchmean_mixed_nontarget_permuted_soft1_on_high_entropy_half_hard_ce_on_low_entropy_half"
            if args.group == "high"
            else "batchmean_mixed_hard_ce_on_high_entropy_half_nontarget_permuted_soft1_on_low_entropy_half"
        ),
        "soft_supervised_images_per_epoch": 50,
        "soft_label_allocation": (
            "top5_calibration_entropy_within_each_class_nontarget_identity_permuted"
            if args.group == "high"
            else "bottom5_calibration_entropy_within_each_class_nontarget_identity_permuted"
        ),
        "primary_metric": "final_top1_at_update_4000",
    }
    errors = [
        f"{key}={payload.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if payload.get(key) != value
    ]
    if Path(payload.get("train_manifest", "")).resolve() != manifest.resolve() or payload.get("train_manifest_sha256") != file_sha256(manifest):
        errors.append("lambda0 manifest was not exactly reused")
    if payload.get("initial_student_state_sha256") != original["initial_student_state_sha256"]:
        errors.append("Student initialization differs from original Soft allocation arm")
    if payload.get("train_manifest_sha256") != original["train_manifest_sha256"]:
        errors.append("manifest differs from original Soft allocation arm")
    if Path(payload.get("non_target_permutation_file", "")).resolve() != permutation_file.resolve() or payload.get("non_target_permutation_file_sha256") != file_sha256(permutation_file):
        errors.append("frozen permutation provenance mismatch")
    history = payload.get("test_history", [])
    if [row.get("epoch") for row in history] != list(EVAL_EPOCHS):
        errors.append("evaluation schedule mismatch")
    for row in history:
        if row.get("update") != 2 * row["epoch"] or not math.isclose(row.get("lr", -1), lr_for_epoch(row["epoch"]), abs_tol=1e-12):
            errors.append(f"epoch/update/LR mismatch at {row.get('epoch')}")
    if not history or payload.get("final_top1") != history[-1].get("test_top1"):
        errors.append("Final mismatch")
    teacher_stats = payload.get("teacher_online_label_statistics", {})
    if teacher_stats.get("views") != 100000:
        errors.append("Teacher view count mismatch")
    for key in ("mean_entropy", "mean_maximum_probability", "mean_true_class_probability", "mean_argmax_matches_true_class"):
        if not math.isclose(teacher_stats.get(key, float("nan")), original["teacher_online_label_statistics"][key], abs_tol=1e-6):
            errors.append(f"Teacher input/statistics changed: {key}")
    invariant = payload.get("permutation_invariant_audit", {})
    if invariant.get("views") != 100000 or invariant.get("batches") != 4000:
        errors.append("permutation invariant audit coverage mismatch")
    for key in (
        "max_abs_entropy_delta",
        "max_abs_true_probability_delta",
        "max_abs_maximum_probability_delta",
        "max_abs_onehot_l1_delta",
        "max_abs_onehot_l2_delta",
    ):
        if float(invariant.get(key, float("inf"))) > 1e-5:
            errors.append(f"permutation failed to preserve {key}")
    if invariant.get("argmax_correctness_mismatches") != 0:
        errors.append("permutation changed Teacher argmax correctness")
    if len(payload.get("per_class_final", [])) != CLASSES:
        errors.append("per-class result missing")
    for key, hash_key in (
        ("protocol_spec", "protocol_spec_sha256"),
        ("student_model_source", "student_model_source_sha256"),
        ("teacher_checkpoint", "teacher_checkpoint_sha256"),
        ("final_checkpoint", "final_checkpoint_sha256"),
    ):
        path = Path(payload.get(key, ""))
        if not path.is_file() or file_sha256(path) != payload.get(hash_key):
            errors.append(f"missing or changed provenance: {key}")
    if errors:
        raise RuntimeError("non-target permutation result audit failed:\n" + "\n".join(errors))
    print(json.dumps({"status": "complete", "result": str(args.result.resolve())}))


if __name__ == "__main__":
    main()
