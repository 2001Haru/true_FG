"""Extract image-level Teacher predictions from the frozen 16 calibration views."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from imagenette_entropy_protocol import (
    CALIBRATION_VIEW_SEED,
    CLASSES,
    SELECTION_VIEWS,
    build_teacher,
    file_sha256,
    normalize,
    validate_official_split,
)
from prepare_imagenette_entropy_selection import EntropyViewDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--per-image-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--workers", default=16, type=int)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    train_dataset, _ = validate_official_split(data_root)
    frozen = json.loads(args.per_image_audit.read_text(encoding="utf-8"))
    frozen_rows = frozen["images"]
    relative_paths = [
        Path(path).relative_to(data_root).as_posix()
        for path, _ in train_dataset.samples
    ]
    if [row["relative_path"] for row in frozen_rows] != relative_paths:
        raise RuntimeError("frozen per-image audit differs from current train split")

    views = EntropyViewDataset(
        train_dataset.samples,
        data_root,
        "imagenette-selection-calibration-view-v1",
        CALIBRATION_VIEW_SEED,
    )
    loader = DataLoader(
        views,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    teacher = build_teacher(args.teacher_checkpoint.resolve()).cuda().eval()
    probability_sum = torch.zeros(len(train_dataset), CLASSES, dtype=torch.float64)
    entropy_sum = torch.zeros(len(train_dataset), dtype=torch.float64)
    view_correct_sum = torch.zeros(len(train_dataset), dtype=torch.float64)
    counts = torch.zeros(len(train_dataset), dtype=torch.int64)
    with torch.inference_mode():
        for batch_index, (images, targets, indices) in enumerate(loader):
            images = normalize(images.cuda(non_blocking=True))
            targets_gpu = targets.cuda(non_blocking=True)
            logits = teacher(images).float()
            log_probabilities = F.log_softmax(logits, dim=1)
            probabilities = log_probabilities.exp()
            entropy = -(probabilities * log_probabilities).sum(dim=1) / torch.log(
                probabilities.new_tensor(float(CLASSES))
            )
            indices = indices.long()
            probability_sum.index_add_(0, indices, probabilities.double().cpu())
            entropy_sum.index_add_(0, indices, entropy.double().cpu())
            view_correct_sum.index_add_(
                0,
                indices,
                probabilities.argmax(dim=1).eq(targets_gpu).double().cpu(),
            )
            counts.index_add_(0, indices, torch.ones_like(indices))
            if batch_index % 50 == 0:
                print(
                    f"views={min((batch_index + 1) * args.batch_size, len(views))}/{len(views)}",
                    flush=True,
                )
    if not torch.equal(counts, torch.full_like(counts, SELECTION_VIEWS)):
        raise RuntimeError("not every image received exactly 16 calibration views")
    mean_probabilities = (probability_sum / counts[:, None]).float()
    mean_view_entropy = (entropy_sum / counts).float()
    view_correct_rate = (view_correct_sum / counts).float()
    frozen_entropy = torch.tensor(
        [row["calibration_entropy"] for row in frozen_rows], dtype=torch.float32
    )
    max_entropy_delta = float((mean_view_entropy - frozen_entropy).abs().max())
    if max_entropy_delta > 2e-6:
        raise RuntimeError(
            f"recomputed calibration entropy differs from frozen audit: {max_entropy_delta}"
        )
    aggregate_prediction = mean_probabilities.argmax(dim=1)
    targets = torch.tensor(train_dataset.targets, dtype=torch.long)
    payload = {
        "status": "complete",
        "definition": {
            "color_entropy": "mean of 16 per-view entropy/log(10), identical to selection statistic",
            "image_prediction": "argmax of the probability vector averaged over the same 16 views",
            "maximum_probability": "maximum of the 16-view mean probability vector",
            "correct": "image_prediction equals official true class",
        },
        "data_root": str(data_root),
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": file_sha256(args.teacher_checkpoint.resolve()),
        "per_image_audit": str(args.per_image_audit.resolve()),
        "per_image_audit_sha256": file_sha256(args.per_image_audit.resolve()),
        "calibration_view_seed": CALIBRATION_VIEW_SEED,
        "calibration_views": SELECTION_VIEWS,
        "relative_paths": relative_paths,
        "targets": targets,
        "mean_probabilities": mean_probabilities,
        "mean_view_normalized_entropy": mean_view_entropy,
        "view_correct_rate": view_correct_rate,
        "aggregate_prediction": aggregate_prediction,
        "aggregate_maximum_probability": mean_probabilities.max(dim=1).values,
        "aggregate_correct": aggregate_prediction.eq(targets),
        "max_abs_entropy_delta_vs_frozen": max_entropy_delta,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "status": "complete",
                "images": len(targets),
                "aggregate_accuracy": float(payload["aggregate_correct"].float().mean()),
                "max_abs_entropy_delta_vs_frozen": max_entropy_delta,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

