import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from config import get_dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def atomic_json_dump(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct = 0
    loss_sum = 0.0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += criterion(logits, labels).item()
        correct += logits.argmax(1).eq(labels).sum().item()
        total += labels.numel()
    return 100.0 * correct / total, loss_sum / total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Train the plain ImageNet-pretrained ResNet18 teacher used by fine-grained SRe2L++"
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=51,
                        help="51 executes official epoch indices 0 through 50")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lr-rate", type=float, default=0.9)
    parser.add_argument("--lr-duration", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = get_dataset(args.dataset_name)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for teacher training")
    if args.epochs != 51:
        print(f"WARNING: epochs={args.epochs}; faithful FD2 execution is 51 (0..50)", flush=True)

    train_dir = args.data_dir / "train"
    test_dir = args.data_dir / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(f"Expected ImageFolder train/test under {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = args.output_dir / "complete.json"
    if args.skip_completed and completion_path.is_file():
        payload = json.loads(completion_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete" and payload.get("seed") == args.seed:
            print(f"Teacher already complete: {completion_path}", flush=True)
            return

    seed_everything(args.seed, args.deterministic)
    device = torch.device("cuda")
    normalize = transforms.Normalize(cfg.mean, cfg.std)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    train_set = datasets.ImageFolder(train_dir, transform=train_transform)
    test_set = datasets.ImageFolder(test_dir, transform=test_transform)
    if len(train_set.classes) != cfg.classes or train_set.class_to_idx != test_set.class_to_idx:
        raise RuntimeError(
            f"Class mismatch: train={len(train_set.classes)}, test={len(test_set.classes)}, "
            f"expected={cfg.classes}"
        )

    generator = torch.Generator().manual_seed(args.seed)
    loader_options = dict(
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=False,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **loader_options,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, cfg.classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: args.lr_rate ** (epoch / args.lr_duration),
    )

    last_checkpoint = args.output_dir / "last_checkpoint.pth.tar"
    start_epoch = 0
    best_accuracy = -1.0
    best_epoch = -1
    if args.resume and last_checkpoint.is_file():
        checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["seed"] != args.seed or checkpoint["dataset"] != cfg.name:
            raise RuntimeError("Resume checkpoint provenance does not match requested run")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_accuracy = checkpoint["best_accuracy"]
        best_epoch = checkpoint["best_epoch"]
        print(f"Resuming at epoch {start_epoch}", flush=True)

    manifest = {
        "status": "running",
        "dataset": cfg.name,
        "dataset_config": cfg.to_dict(),
        "data_dir": str(args.data_dir.resolve()),
        "train_images": len(train_set),
        "test_images": len(test_set),
        "classes": train_set.classes,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": "SGD",
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "lr_schedule": f"{args.lr_rate}^(epoch/{args.lr_duration})",
        "initial_weights": "torchvision ResNet18 IMAGENET1K_V1",
        "augmentation": "HorizontalFlip+ColorJitter(0.2,0.2,0.2,0.1)+Rotation(15deg)",
        "deterministic": args.deterministic,
        "git_revision": git_revision(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "python": platform.python_version(),
        "argv": sys.argv,
    }
    atomic_json_dump(manifest, args.output_dir / "manifest.json")
    metrics_path = args.output_dir / "metrics.jsonl"

    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        model.train()
        loss_sum = 0.0
        correct = 0
        total = 0
        lr = optimizer.param_groups[0]["lr"]
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.numel()
            correct += logits.argmax(1).eq(labels).sum().item()
            total += labels.numel()

        validation_accuracy, validation_loss = evaluate(model, test_loader, device)
        record = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": loss_sum / total,
            "train_accuracy": 100.0 * correct / total,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "seconds": time.time() - started,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            atomic_torch_save(model.state_dict(), args.output_dir / "ResNet18.pth")
            atomic_json_dump(
                {"epoch": best_epoch, "validation_accuracy": best_accuracy, "seed": args.seed},
                args.output_dir / "best.json",
            )

        scheduler.step()
        atomic_torch_save(
            {
                "epoch": epoch,
                "dataset": cfg.name,
                "seed": args.seed,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_accuracy": best_accuracy,
                "best_epoch": best_epoch,
            },
            last_checkpoint,
        )

    atomic_torch_save(model.state_dict(), args.output_dir / "ResNet18_epoch050.pth")
    completion = {
        **manifest,
        "status": "complete",
        "best_validation_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "final_epoch": args.epochs - 1,
    }
    atomic_json_dump(completion, completion_path)
    print(
        f"Teacher complete: best_top1={best_accuracy:.4f} epoch={best_epoch} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
