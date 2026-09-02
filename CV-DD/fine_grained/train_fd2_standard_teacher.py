"""Train Backbone+CAL under the project's standard protocol.

Only the explicitly selected FD2 interpretation varies. Dataset preparation,
ImageNet-1K initialization, optimizer, schedule, augmentation and loader policy
remain identical to the standard plain-teacher baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

from config import get_dataset
from fd2_standard_components import (
    CAL,
    CenterLoss,
    LastFeatureHook,
    batch_augment,
    dataset_setting,
    semantic_setting,
    update_feature_centers,
)


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_save(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int, seed: int) -> None:
    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_loaders(data_dir: Path, cfg, batch_size: int, workers: int, seed: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)]
    )
    train_set = torchvision.datasets.ImageFolder(data_dir / "train", transform=train_transform)
    test_set = torchvision.datasets.ImageFolder(data_dir / "test", transform=test_transform)
    common = dict(
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=partial(seed_worker, seed=seed),
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=False, **common
    )
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, drop_last=False, **common)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(backbone, cal, hook, loader, criterion, device):
    backbone.eval()
    cal.eval()
    count = 0
    backbone_correct = 0
    cal_correct = 0
    backbone_loss = 0.0
    cal_loss = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = backbone(images)
        raw, _effect, _features, _attention, _maps = cal(hook.feature)
        count += labels.numel()
        backbone_correct += (logits.argmax(1) == labels).sum().item()
        cal_correct += (raw.argmax(1) == labels).sum().item()
        backbone_loss += criterion(logits, labels).item()
        cal_loss += criterion(raw, labels).item()
    return {
        "backbone_accuracy": 100.0 * backbone_correct / count,
        "cal_accuracy": 100.0 * cal_correct / count,
        "backbone_loss": backbone_loss / len(loader),
        "cal_loss": cal_loss / len(loader),
    }


def parse_args():
    parser = argparse.ArgumentParser("Standard-protocol FD2 Backbone+CAL teacher")
    parser.add_argument("--semantics", required=True, choices=("released_semantics", "paper_literal"))
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eta-min", type=float, default=1e-5)
    parser.add_argument("--center-weight", type=float, default=1.0)
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    cfg = get_dataset(args.dataset_name)
    fd2_cfg = dataset_setting(cfg.name)
    semantics = semantic_setting(args.semantics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = args.output_dir / "complete.json"
    if args.skip_completed and completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        expected = {
            "status": "complete",
            "semantics": args.semantics,
            "dataset": cfg.name,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "dataloader_workers": args.workers,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "eta_min": args.eta_min,
            "center_weight_eta": args.center_weight,
        }
        if all(completion.get(key) == value for key, value in expected.items()):
            print(f"FD2 teacher already complete: {completion_path}")
            return
        raise RuntimeError(f"Existing FD2 teacher has incompatible protocol: {completion_path}")

    seed_everything(args.seed)
    device = torch.device("cuda")
    train_loader, test_loader = make_loaders(args.data_dir, cfg, args.batch_size, args.workers, args.seed)
    backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Linear(backbone.fc.in_features, cfg.classes)
    backbone.to(device)
    cal = CAL(cfg.classes, fd2_cfg.attention_maps).to(device)
    hook = LastFeatureHook(backbone.layer4[1].bn2)
    feature_centers = torch.zeros(
        cfg.classes, fd2_cfg.attention_maps * cal.num_features, device=device
    )
    criterion = nn.CrossEntropyLoss().to(device)
    center_loss = CenterLoss().to(device)
    optimizer = torch.optim.SGD(
        list(backbone.parameters()) + list(cal.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.eta_min
    )

    manifest = {
        "status": "running",
        "method": "FD2 on standard SRe2L++ foundation",
        "semantics": args.semantics,
        "semantic_setting": semantics.to_dict(),
        "paper_reference": "arXiv:2603.25144v2 (2026-06-27)",
        "released_reference": [
            "FD2/squeeze/squeeze_cal.py",
            "FD2/models/cal.py",
        ],
        "dataset": cfg.name,
        "seed": args.seed,
        "initial_weights": "torchvision ResNet18 IMAGENET1K_V1",
        "architecture": "joint ResNet18 backbone + CAL",
        "attention_maps": fd2_cfg.attention_maps,
        "cal_ratio": fd2_cfg.cal_ratio,
        "center_weight_eta": args.center_weight,
        "prototype_momentum": semantics.prototype_momentum,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "dataloader_workers": args.workers,
        "persistent_workers": False,
        "optimizer": "SGD",
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR",
        "eta_min": args.eta_min,
        "augmentation": "HorizontalFlip+ColorJitter(0.2,0.2,0.2,0.1)+Rotation(15deg)",
        "checkpoint_selection": "final epoch",
        "backbone_soft_label_source": "ResNet18.pth",
    }
    atomic_json(manifest, args.output_dir / "manifest.json")
    metrics_path = args.output_dir / "metrics.jsonl"
    final_metrics = None

    for epoch in range(args.epochs):
        started = time.time()
        backbone.train()
        cal.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            backbone_logits = backbone(images)
            raw, effect, features, attention, _attention_maps = cal(hook.feature)
            centers_for_loss = F.normalize(feature_centers[labels], dim=-1)
            update_feature_centers(
                feature_centers, features, labels, args.semantics, semantics.prototype_momentum
            )
            backbone_term = (1.0 - fd2_cfg.cal_ratio) * criterion(backbone_logits, labels)
            if args.semantics == "released_semantics":
                with torch.no_grad():
                    crop = batch_augment(images, attention[:, :1], "crop", (0.4, 0.6), 0.1)
                    drop = batch_augment(images, attention[:, 1:], "drop", (0.2, 0.5))
                augmented = torch.cat((crop, drop))
                augmented_labels = torch.cat((labels, labels))
                backbone(augmented)
                raw_aug, effect_aug, _features_aug, _attention_aug, _maps_aug = cal(hook.feature)
                effect_aux = torch.cat((effect, effect_aug))
                labels_aux = torch.cat((labels, augmented_labels))
                cal_term = fd2_cfg.cal_ratio * (
                    criterion(raw, labels) / 3.0
                    + criterion(effect_aux, labels_aux)
                    + criterion(raw_aug, augmented_labels) * 2.0 / 3.0
                    + args.center_weight * center_loss(features, centers_for_loss)
                )
            else:
                cal_term = fd2_cfg.cal_ratio * (
                    criterion(raw, labels)
                    + criterion(effect, labels)
                    + args.center_weight * center_loss(features, centers_for_loss)
                )
            loss = backbone_term + cal_term
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            train_metrics = evaluate(backbone, cal, hook, train_loader, criterion, device)
            validation_metrics = evaluate(backbone, cal, hook, test_loader, criterion, device)
            record = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "seconds": time.time() - started,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            if epoch == args.epochs - 1:
                final_metrics = record
        scheduler.step()

    joint = {
        "ResNet18": backbone.state_dict(),
        "cal": cal.state_dict(),
        "feature_center": feature_centers.cpu(),
        "semantics": args.semantics,
        "dataset": cfg.name,
        "attention_maps": fd2_cfg.attention_maps,
        "cal_ratio": fd2_cfg.cal_ratio,
    }
    atomic_save(joint, args.output_dir / "FD2_ResNet18_CAL.pth")
    atomic_save(backbone.state_dict(), args.output_dir / "ResNet18.pth")
    if final_metrics is None:
        raise RuntimeError("Final FD2 teacher checkpoint was not evaluated")
    completion = {
        **manifest,
        "status": "complete",
        "selected_epoch": args.epochs - 1,
        "selected_metrics": final_metrics,
    }
    atomic_json(completion, completion_path)
    hook.close()
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
