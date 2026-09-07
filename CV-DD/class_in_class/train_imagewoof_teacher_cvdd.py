"""Train one ImageWoof ResNet18 with CV-DD squeeze_imagenette semantics.

The optimization path matches the released CV-DD ResNet18 teacher: random
initialization, RRC224+flip, batch 64, SGD 0.01/0.9/1e-4, 300 epochs, and
CosineAnnealingLR(T_max=100).  The upstream script is unseeded; this wrapper
preserves that behavior while recording provenance and the complete history.
"""

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


EXPECTED_TRAIN = 9025
EXPECTED_TEST = 3929
EXPECTED_CLASSES = (
    "n02086240", "n02087394", "n02088364", "n02089973", "n02093754",
    "n02096294", "n02099601", "n02105641", "n02111889", "n02115641",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = correct = total = 0.0
    for inputs, labels in loader:
        inputs = inputs.to(device); labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward(); optimizer.step()
        loss_sum += float(loss) * len(labels)
        correct += int(outputs.argmax(1).eq(labels).sum())
        total += len(labels)
    return {"loss": loss_sum / total, "top1": 100.0 * correct / total}


@torch.inference_mode()
def validate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0.0
    class_correct = torch.zeros(len(EXPECTED_CLASSES), dtype=torch.int64)
    class_total = torch.zeros(len(EXPECTED_CLASSES), dtype=torch.int64)
    for inputs, labels in loader:
        inputs = inputs.to(device); labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        predictions = outputs.argmax(1)
        loss_sum += float(loss) * len(labels)
        correct += int(predictions.eq(labels).sum())
        total += len(labels)
        class_total.index_add_(0, labels.cpu(), torch.ones_like(labels.cpu()))
        class_correct.index_add_(0, labels.cpu(), predictions.eq(labels).cpu().long())
    return {
        "loss": loss_sum / total, "top1": 100.0 * correct / total,
        "per_class_top1": (100.0 * class_correct.double() / class_total).tolist(),
        "per_class_counts": class_total.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    data = args.data_dir.resolve(); output = args.output_dir.resolve()
    checkpoint = output / "ResNet18.pth"
    if output.exists():
        raise RuntimeError(f"refusing existing output directory: {output}")
    output.mkdir(parents=True)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_set = datasets.ImageFolder(data / "train", transform=train_transform)
    test_set = datasets.ImageFolder(data / "test", transform=test_transform)
    if len(train_set) != EXPECTED_TRAIN or len(test_set) != EXPECTED_TEST:
        raise RuntimeError(f"ImageWoof2 counts={len(train_set)}/{len(test_set)}")
    if tuple(train_set.classes) != EXPECTED_CLASSES or train_set.class_to_idx != test_set.class_to_idx:
        raise RuntimeError("ImageWoof2 class mapping mismatch")
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=4)

    initial_torch_seed = torch.initial_seed()
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(EXPECTED_CLASSES))
    device = torch.device("cuda")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)
    history = []
    best = {"epoch": None, "top1": float("-inf")}
    started = time.time()
    for epoch in range(1, 301):
        lr = float(optimizer.param_groups[0]["lr"])
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        test_metrics = validate(model, test_loader, criterion, device)
        row = {"epoch": epoch, "lr": lr, "train": train_metrics, "test": test_metrics}
        history.append(row)
        if test_metrics["top1"] > best["top1"]:
            best = {"epoch": epoch, "top1": test_metrics["top1"]}
        print(json.dumps({"epoch": epoch, "lr": lr, "train": train_metrics, "test_top1": test_metrics["top1"]}), flush=True)
        scheduler.step()
    torch.save(model.state_dict(), checkpoint)
    resource_manifest = data / "resource_manifest.json"
    payload = {
        "status": "complete", "protocol": "cvdd_squeeze_imagenette_resnet18_applied_to_imagewoof2",
        "released_source": "CV-DD/squeeze/squeeze_imagenette.py",
        "intentional_changes": [
            "dataset changed from ImageNette2 to ImageWoof2",
            "checkpoint filename normalized to ResNet18.pth",
            "complete history and provenance recorded; logging cadence changed without numerical effect"
        ],
        "unseeded_release_semantics": True, "torch_initial_seed_observed": initial_torch_seed,
        "model": "torchvision ResNet18", "initialization": "random", "classes": len(EXPECTED_CLASSES),
        "class_to_idx": train_set.class_to_idx, "train_images": len(train_set), "test_images": len(test_set),
        "train_transform": "RandomResizedCrop224 + HorizontalFlip + ToTensor + ImageNet normalization",
        "test_transform": "Resize256 + CenterCrop224 + ToTensor + ImageNet normalization",
        "batch_size": 64, "workers": 4, "persistent_workers": False,
        "optimizer": "SGD", "initial_lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingLR(T_max=100, eta_min=0) stepped after each epoch",
        "epochs": 300, "checkpoint_selection": "final epoch only",
        "history": history, "best_test_top1_diagnostic": best,
        "final_train": history[-1]["train"], "final_test": history[-1]["test"],
        "data_root": str(data), "resource_manifest": str(resource_manifest),
        "resource_manifest_sha256": sha256(resource_manifest),
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "elapsed_seconds": time.time() - started,
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
    }
    atomic_json(output / "training_history.json", payload)
    print(json.dumps({"status": "complete", "checkpoint": str(checkpoint), "final_test": payload["final_test"], "best": best}, indent=2))


if __name__ == "__main__":
    main()
