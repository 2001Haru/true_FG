"""Strict result and checkpoint audit for the frozen hard-label v1 protocol."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from hard_label_v1_protocol import (
    BACKBONE_LR,
    BACKBONE_MIN_LR,
    EVAL_EVERY_UPDATES,
    GRADIENT_ACCUMULATION_STEPS,
    HEAD_LR,
    HEAD_MIN_LR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MOMENTUM,
    PHYSICAL_BATCH_SIZE,
    PROTOCOL_NAME,
    TOTAL_UPDATES,
    TRAIN_WORKERS,
    VALIDATION_BATCH_SIZE,
    WEIGHT_DECAY,
    cosine_lr,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same(actual, expected) -> bool:
    if isinstance(expected, float):
        return actual is not None and math.isclose(float(actual), expected, abs_tol=1e-12)
    return actual == expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--classes", required=True, type=int)
    parser.add_argument("--ipc", required=True, type=int)
    parser.add_argument("--student-seed", required=True, type=int)
    parser.add_argument("--validation-images", required=True, type=int)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    protocol_spec = Path(__file__).resolve().with_name("hard_label_v1_protocol.json")
    expected = {
        "status": "complete",
        "protocol": PROTOCOL_NAME,
        "protocol_spec_sha256": sha256(protocol_spec),
        "dataset": args.dataset,
        "ipc": args.ipc,
        "training_target": "hard_coarse_label",
        "loss": "cross_entropy",
        "label_smoothing": 0.0,
        "cutmix": False,
        "mixup": False,
        "student_model": "torchvision_resnet18",
        "student_initialization": "imagenet1k_v1",
        "classification_head_initialization": "torch_nn_linear_default_random",
        "backbone_trainable_from_update": 1,
        "student_seed": args.student_seed,
        "optimizer": "sgd",
        "momentum": MOMENTUM,
        "dampening": 0.0,
        "nesterov": False,
        "weight_decay": WEIGHT_DECAY,
        "backbone_initial_lr": BACKBONE_LR,
        "head_initial_lr": HEAD_LR,
        "backbone_min_lr": BACKBONE_MIN_LR,
        "head_min_lr": HEAD_MIN_LR,
        "scheduler": "per_update_full_cosine",
        "warmup_updates": 0,
        "total_optimizer_updates": TOTAL_UPDATES,
        "updates_completed": TOTAL_UPDATES,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "physical_batch_size": PHYSICAL_BATCH_SIZE,
        "drop_last": False,
        "bn_running_statistics": "updated_in_train_mode",
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
        "image_size": IMAGE_SIZE,
        "train_images": args.classes * args.ipc,
        "validation_images": args.validation_images,
        "num_classes": args.classes,
        "dataloader_workers": TRAIN_WORKERS,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "validation_batch_size": VALIDATION_BATCH_SIZE,
        "evaluation_every_updates": EVAL_EVERY_UPDATES,
        "primary_metric": "final_update_top1",
        "scheduler_endpoint_backbone_lr": BACKBONE_MIN_LR,
        "scheduler_endpoint_head_lr": HEAD_MIN_LR,
    }
    errors = []
    for key, value in expected.items():
        if not same(payload.get(key), value):
            errors.append(f"{key}={payload.get(key)!r}, expected {value!r}")
    history = payload.get("validation_history", [])
    expected_updates = list(range(EVAL_EVERY_UPDATES, TOTAL_UPDATES + 1, EVAL_EVERY_UPDATES))
    if [row.get("update") for row in history] != expected_updates:
        errors.append("validation update schedule differs from frozen protocol")
    for row in history:
        t = int(row["update"]) - 1
        for key, initial, minimum in (
            ("backbone_lr_used", BACKBONE_LR, BACKBONE_MIN_LR),
            ("head_lr_used", HEAD_LR, HEAD_MIN_LR),
        ):
            expected_lr = cosine_lr(initial, minimum, t, TOTAL_UPDATES)
            if not same(row.get(key), expected_lr):
                errors.append(f"update {row['update']}: {key} schedule mismatch")
    if not history or not same(payload.get("final_top1"), history[-1].get("top1")):
        errors.append("final_top1 does not match the U=3000 validation")
    per_class = payload.get("per_class_final", [])
    if len(per_class) != args.classes:
        errors.append(f"per-class rows {len(per_class)} != {args.classes}")
    if sum(int(row.get("total", 0)) for row in per_class) != args.validation_images:
        errors.append("per-class validation total mismatch")
    checkpoint = Path(payload.get("final_checkpoint", ""))
    if not checkpoint.is_file():
        errors.append(f"missing final checkpoint: {checkpoint}")
    elif sha256(checkpoint) != payload.get("final_checkpoint_sha256"):
        errors.append("final checkpoint SHA-256 mismatch")
    if Path(payload.get("protocol_spec", "")).resolve() != protocol_spec.resolve():
        errors.append("protocol specification path mismatch")
    names = payload.get("optimizer_parameter_names", {})
    if any("bn" in name for name in names.get("backbone_decay", [])):
        errors.append("a BN parameter appears in a weight-decay group")
    if any(name.endswith("bias") for name in names.get("backbone_decay", []) + names.get("head_decay", [])):
        errors.append("a bias appears in a weight-decay group")
    if errors:
        raise RuntimeError("hard-label v1 audit failed:\n" + "\n".join(errors))
    print(
        json.dumps(
            {
                "status": "complete",
                "result": str(args.result.resolve()),
                "final_top1": payload["final_top1"],
                "updates": payload["updates_completed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
