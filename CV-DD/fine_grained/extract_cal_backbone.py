import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from config import get_dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct = 0
    total = 0
    loss_sum = 0.0
    logit_std_sum = 0.0
    entropy_sums = {1.0: 0.0, 3.0: 0.0, 20.0: 0.0}
    for images, labels in loader:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        logits = model(images)
        batch = labels.numel()
        loss_sum += criterion(logits, labels).item()
        correct += logits.argmax(1).eq(labels).sum().item()
        total += batch
        logit_std_sum += logits.std(1, unbiased=False).sum().item()
        for temperature in entropy_sums:
            probabilities = torch.softmax(logits / temperature, dim=1)
            entropy = -(
                probabilities * probabilities.clamp_min(1e-12).log()
            ).sum(1)
            entropy_sums[temperature] += entropy.sum().item()
    classes = logits.shape[1]
    return {
        "validation_images": total,
        "validation_accuracy": 100.0 * correct / total,
        "validation_cross_entropy": loss_sum / total,
        "mean_per_image_logit_std": logit_std_sum / total,
        "mean_normalized_entropy": {
            str(int(temperature)): value / total / math.log(classes)
            for temperature, value in entropy_sums.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        "Extract and audit a plain ResNet18 backbone from an FD2 CAL checkpoint"
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-key", default="ResNet18")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--attention-maps", type=int, required=True)
    parser.add_argument("--cal-ratio", type=float, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for teacher evaluation")
    cfg = get_dataset(args.dataset_name)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    test_dir = args.data_dir / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(test_dir)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or args.checkpoint_key not in payload:
        raise RuntimeError(
            f"CAL checkpoint does not contain key {args.checkpoint_key!r}: "
            f"{args.checkpoint}"
        )
    state_dict = payload[args.checkpoint_key]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, cfg.classes)
    model.load_state_dict(state_dict, strict=True)
    model.cuda()

    dataset = datasets.ImageFolder(
        test_dir,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ]),
    )
    if len(dataset.classes) != cfg.classes:
        raise RuntimeError(
            f"validation classes={len(dataset.classes)} != expected {cfg.classes}"
        )
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    metrics = evaluate(model, loader)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint = args.output_dir / "ResNet18.pth"
    atomic_torch_save(state_dict, output_checkpoint)
    completion = {
        "status": "complete",
        "dataset": cfg.name,
        "teacher_variant": "joint_cal_pretraining_extracted_backbone",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_key": args.checkpoint_key,
        "attention_maps": args.attention_maps,
        "cal_ratio": args.cal_ratio,
        "best_validation_accuracy": metrics["validation_accuracy"],
        "best_epoch": -1,
        "extracted_checkpoint_sha256": sha256(output_checkpoint),
        "metrics": metrics,
    }
    atomic_json_dump(completion, args.output_dir / "complete.json")
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
