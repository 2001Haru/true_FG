"""Frozen constants and pure helpers for the project hard-label v1 protocol."""

import math


PROTOCOL_NAME = "hard_label_v1"
TOTAL_UPDATES = 3000
PHYSICAL_BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 1
BACKBONE_LR = 3e-4
HEAD_LR = 3e-3
BACKBONE_MIN_LR = 0.0
HEAD_MIN_LR = 0.0
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EVAL_EVERY_UPDATES = 300
TRAIN_WORKERS = 8
VALIDATION_BATCH_SIZE = 256
IMAGE_SIZE = 224
RESIZE_SIZE = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def cosine_lr(initial_lr: float, minimum_lr: float, t: int, total_updates: int) -> float:
    """Return eta_min + (eta_0-eta_min)/2 * (1+cos(pi*t/U))."""
    if total_updates <= 0:
        raise ValueError("total_updates must be positive")
    if not 0 <= t <= total_updates:
        raise ValueError(f"schedule position t={t} outside [0, {total_updates}]")
    if initial_lr < 0 or minimum_lr < 0 or minimum_lr > initial_lr:
        raise ValueError("learning rates must satisfy 0 <= minimum_lr <= initial_lr")
    return minimum_lr + 0.5 * (initial_lr - minimum_lr) * (
        1.0 + math.cos(math.pi * t / total_updates)
    )

