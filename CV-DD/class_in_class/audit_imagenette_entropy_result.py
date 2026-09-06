"""Strict audit for one ImageNette entropy-selection student result."""

import argparse
import json
import math
from pathlib import Path

from imagenette_entropy_protocol import (
    BATCH_SIZE,
    CLASSES,
    EVAL_EVERY_EPOCHS,
    IPC,
    LR,
    MOMENTUM,
    TEST_IMAGES,
    TOTAL_UPDATES,
    TRAIN_EPOCHS,
    TRAIN_SIZE,
    WEIGHT_DECAY,
    file_sha256,
    lr_for_epoch,
)


def same(actual, expected):
    if isinstance(expected, float):
        return actual is not None and math.isclose(float(actual), expected, abs_tol=1e-12)
    return actual == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--lambda", dest="lambda_value", required=True, type=int)
    parser.add_argument("--selection-seed", required=True, type=int)
    parser.add_argument("--student-seed", required=True, type=int)
    parser.add_argument("--supervision", choices=("hard", "soft1", "kd4"), required=True)
    parser.add_argument("--protocol-name", default="imagenette_entropy_selection_v1")
    parser.add_argument("--protocol-spec", type=Path)
    parser.add_argument("--expected-eval-epochs", nargs="+", type=int)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    protocol_spec = (
        args.protocol_spec.resolve()
        if args.protocol_spec is not None
        else Path(__file__).resolve().with_name(
            "imagenette_entropy_selection_protocol.json"
        )
    )
    expected = {
        "status": "complete",
        "protocol": args.protocol_name,
        "protocol_spec_sha256": file_sha256(protocol_spec),
        "dataset": "imagenet-nette",
        "classes": CLASSES,
        "ipc": IPC,
        "train_images": TRAIN_SIZE,
        "test_images": TEST_IMAGES,
        "lambda": args.lambda_value,
        "selection_seed": args.selection_seed,
        "student_seed": args.student_seed,
        "supervision": args.supervision,
        "student_model": "CoDA_released_ResNetAP10",
        "student_initialization": "random",
        "normalization_layer": "GroupNorm(C,C)_released_instance_semantics",
        "image_size": 256,
        "optimizer": "SGD",
        "initial_lr": LR,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "weight_decay_exclusions": [],
        "batch_size": BATCH_SIZE,
        "actual_batch_sizes_per_epoch": [64, 36],
        "loss_batch_denominator": "current_actual_batch_size",
        "gradient_accumulation_steps": 1,
        "drop_last": False,
        "epochs": TRAIN_EPOCHS,
        "updates_completed": TOTAL_UPDATES,
        "scheduler": "released_multistep_epoch",
        "lr_milestones_after_epochs": [1333, 1666],
        "lr_gamma": 0.2,
        "warmup": False,
        "label_smoothing": 0.0,
        "cutmix": False,
        "mixup": False,
        "training_sample_weighting": "equal",
        "primary_metric": "final_top1_at_update_4000",
    }
    errors = [
        f"{key}={payload.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if not same(payload.get(key), value)
    ]
    expected_loss = {
        "hard": "mean_cross_entropy_true_one_hot",
        "soft1": "mean_soft_target_cross_entropy",
        "kd4": "16_times_KL_teacher_to_student_batchmean",
    }[args.supervision]
    if payload.get("loss") != expected_loss:
        errors.append("loss definition mismatch")
    teacher_expected = args.supervision != "hard"
    if payload.get("teacher_and_student_view_pixels_identical") != teacher_expected:
        errors.append("same-view Teacher flag mismatch")
    if payload.get("teacher_mode") != ("eval_frozen_no_grad" if teacher_expected else None):
        errors.append("Teacher state mismatch")
    expected_teacher_temperature = None if args.supervision == "hard" else (1.0 if args.supervision == "soft1" else 4.0)
    if not same(payload.get("teacher_temperature"), expected_teacher_temperature):
        errors.append("Teacher temperature mismatch")
    if not same(payload.get("student_temperature"), 4.0 if args.supervision == "kd4" else 1.0):
        errors.append("Student temperature mismatch")
    if payload.get("temperature_squared_multiplier") != (args.supervision == "kd4"):
        errors.append("T-squared multiplier mismatch")
    history = payload.get("test_history", [])
    expected_epochs = (
        sorted(args.expected_eval_epochs)
        if args.expected_eval_epochs is not None
        else list(range(EVAL_EVERY_EPOCHS, TRAIN_EPOCHS + 1, EVAL_EVERY_EPOCHS))
    )
    if payload.get("evaluation_epochs", expected_epochs) != expected_epochs:
        errors.append("recorded evaluation epochs mismatch")
    if [row.get("epoch") for row in history] != expected_epochs:
        errors.append("test history cadence mismatch")
    for row in history:
        if row.get("update") != 2 * row["epoch"] or not same(row.get("lr"), lr_for_epoch(row["epoch"])):
            errors.append(f"epoch/update/LR mismatch at {row.get('epoch')}")
    if not history or not same(payload.get("final_top1"), history[-1].get("test_top1")):
        errors.append("Final does not match epoch2000/update4000 history")
    label_stats = payload.get("teacher_online_label_statistics")
    if teacher_expected:
        if not isinstance(label_stats, dict) or label_stats.get("views") != TRAIN_SIZE * TRAIN_EPOCHS:
            errors.append("Teacher online label view count mismatch")
        elif any(
            not math.isfinite(float(label_stats.get(key, float("nan"))))
            for key in (
                "mean_entropy",
                "mean_maximum_probability",
                "mean_true_class_probability",
                "mean_argmax_matches_true_class",
            )
        ):
            errors.append("Teacher online label statistics invalid")
    elif label_stats is not None:
        errors.append("Hard result unexpectedly has Teacher label statistics")
    per_class = payload.get("per_class_final", [])
    if len(per_class) != CLASSES or sum(row.get("total", 0) for row in per_class) != TEST_IMAGES:
        errors.append("per-class test audit mismatch")
    for key, hash_key in (
        ("protocol_spec", "protocol_spec_sha256"),
        ("student_model_source", "student_model_source_sha256"),
        ("train_manifest", "train_manifest_sha256"),
        ("teacher_checkpoint", "teacher_checkpoint_sha256"),
        ("final_checkpoint", "final_checkpoint_sha256"),
    ):
        path = Path(payload.get(key, ""))
        if not path.is_file() or file_sha256(path) != payload.get(hash_key):
            errors.append(f"missing or changed provenance: {key}")
    manifest = json.loads(Path(payload.get("train_manifest", "")).read_text(encoding="utf-8"))
    if manifest.get("teacher_checkpoint_sha256") != payload.get("teacher_checkpoint_sha256"):
        errors.append("training Teacher differs from entropy-selection Teacher")
    split_files = Path(manifest.get("official_split_file_list", ""))
    if (
        not split_files.is_file()
        or file_sha256(split_files) != manifest.get("official_split_file_list_sha256")
    ):
        errors.append("official train/test file list is missing or changed")
    if errors:
        raise RuntimeError("ImageNette entropy result audit failed:\n" + "\n".join(errors))
    print(json.dumps({"status": "complete", "result": str(args.result.resolve())}))


if __name__ == "__main__":
    main()
