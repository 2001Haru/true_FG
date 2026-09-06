"""Strict result audit for lambda0 entropy allocation and deterministic Top10 arms."""

import argparse
import json
import math
from pathlib import Path

from imagenette_entropy_protocol import (
    BATCH_SIZE,
    CLASSES,
    TEST_IMAGES,
    TOTAL_UPDATES,
    TRAIN_EPOCHS,
    TRAIN_SIZE,
    file_sha256,
    lr_for_epoch,
)


EVAL_EPOCHS = (200, 400, 600, 800, 1000, 1200, 1333, 1400, 1600, 1666, 1800, 2000)


def same(actual, expected):
    if isinstance(expected, float):
        return actual is not None and math.isclose(float(actual), expected, abs_tol=1e-12)
    return actual == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument(
        "--arm",
        choices=("high_entropy_soft1", "low_entropy_soft1", "top10_hard", "top10_soft1"),
        required=True,
    )
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument("--student-seed", choices=(42, 43, 44), required=True, type=int)
    parser.add_argument("--experiment-root", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    root = args.experiment_root.resolve()
    spec = Path(__file__).resolve().with_name("imagenette_entropy_allocation_protocol.json")
    is_top = args.arm.startswith("top10_")
    supervision = args.arm.removeprefix("top10_") if is_top else args.arm
    expected_manifest = (
        root / "manifests" / "highest_entropy_top10_per_class.json"
        if is_top
        else root / "manifests" / f"lambda_{0:+d}_rseed{args.selection_seed}.json"
    )
    expected = {
        "status": "complete",
        "protocol": "imagenette_entropy_allocation_v1",
        "protocol_spec_sha256": file_sha256(spec),
        "dataset": "imagenet-nette",
        "classes": CLASSES,
        "ipc": 10,
        "train_images": TRAIN_SIZE,
        "test_images": TEST_IMAGES,
        "student_seed": args.student_seed,
        "supervision": supervision,
        "student_model": "CoDA_released_ResNetAP10",
        "student_initialization": "random",
        "batch_size": BATCH_SIZE,
        "actual_batch_sizes_per_epoch": [64, 36],
        "loss_batch_denominator": "current_actual_batch_size",
        "gradient_accumulation_steps": 1,
        "drop_last": False,
        "epochs": TRAIN_EPOCHS,
        "updates_completed": TOTAL_UPDATES,
        "cutmix": False,
        "mixup": False,
        "primary_metric": "final_top1_at_update_4000",
        "evaluation_epochs": list(EVAL_EPOCHS),
        "evaluation_count": len(EVAL_EPOCHS),
        "teacher_and_student_view_pixels_identical": supervision != "hard",
        "teacher_mode": None if supervision == "hard" else "eval_frozen_no_grad",
        "teacher_temperature": None if supervision == "hard" else 1.0,
        "student_temperature": 1.0,
        "temperature_squared_multiplier": False,
        "soft_supervised_images_per_epoch": {
            "hard": 0,
            "soft1": 100,
            "high_entropy_soft1": 50,
            "low_entropy_soft1": 50,
        }[supervision],
        "soft_label_allocation": {
            "hard": "none",
            "soft1": "all_images",
            "high_entropy_soft1": "top5_calibration_entropy_within_each_class",
            "low_entropy_soft1": "bottom5_calibration_entropy_within_each_class",
        }[supervision],
    }
    errors = [
        f"{key}={payload.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if not same(payload.get(key), value)
    ]
    if Path(payload.get("train_manifest", "")).resolve() != expected_manifest.resolve():
        errors.append("wrong training manifest")
    if file_sha256(expected_manifest) != payload.get("train_manifest_sha256"):
        errors.append("training manifest provenance mismatch")
    history = payload.get("test_history", [])
    if [row.get("epoch") for row in history] != list(EVAL_EPOCHS):
        errors.append("12-point evaluation schedule mismatch")
    for row in history:
        if row.get("update") != 2 * row["epoch"] or not same(row.get("lr"), lr_for_epoch(row["epoch"])):
            errors.append(f"epoch/update/LR mismatch at {row.get('epoch')}")
    if not history or not same(payload.get("final_top1"), history[-1].get("test_top1")):
        errors.append("Final does not match epoch2000 history")
    stats = payload.get("teacher_online_label_statistics")
    expected_views = expected["soft_supervised_images_per_epoch"] * TRAIN_EPOCHS
    if expected_views:
        if not isinstance(stats, dict) or stats.get("views") != expected_views:
            errors.append("Teacher online view count mismatch")
    elif stats is not None:
        errors.append("Hard arm unexpectedly recorded Teacher labels")
    if len(payload.get("per_class_final", [])) != CLASSES:
        errors.append("per-class final audit missing")
    for key, hash_key in (
        ("protocol_spec", "protocol_spec_sha256"),
        ("student_model_source", "student_model_source_sha256"),
        ("teacher_checkpoint", "teacher_checkpoint_sha256"),
        ("final_checkpoint", "final_checkpoint_sha256"),
    ):
        path = Path(payload.get(key, ""))
        if not path.is_file() or file_sha256(path) != payload.get(hash_key):
            errors.append(f"missing or changed provenance: {key}")
    reference_seed = args.selection_seed if args.selection_seed is not None else 0
    reference = root / "results" / "lambda_+0" / f"rseed{reference_seed}" / f"hard_sseed{args.student_seed}.json"
    reference_payload = json.loads(reference.read_text(encoding="utf-8"))
    if payload.get("initial_student_state_sha256") != reference_payload["initial_student_state_sha256"]:
        errors.append("Student initialization differs from paired lambda0 result")
    if not is_top and payload.get("train_manifest_sha256") != reference_payload["train_manifest_sha256"]:
        errors.append("allocation arm does not exactly reuse lambda0 manifest")
    if errors:
        raise RuntimeError("allocation result audit failed:\n" + "\n".join(errors))
    print(json.dumps({"status": "complete", "result": str(args.result.resolve())}))


if __name__ == "__main__":
    main()
