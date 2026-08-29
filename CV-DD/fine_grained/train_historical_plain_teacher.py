import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "FD2"))
from models.utils_models import load_model  # noqa: E402
from squeeze.utils_squeeze import evaluate_loader, load_dataset  # noqa: E402

from config import get_dataset  # noqa: E402


SOURCE_COMMIT = "e07285bd74d3e3ea6398e823f5090018d75e9924"
SOURCE_TRAINER_BLOB = "f0191c72ce8ccb00e5fc29428cd00253c9890d07"
SOURCE_LAUNCHER_BLOB = "5f5ca381cfc90b5fedb6aa728a611755f8787002"


def atomic_json_dump(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(payload: object, path: Path) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        "Reproduce the plain FD2 teacher launcher deleted in commit 908d10f"
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--eta-min", type=float, default=1e-5)
    parser.add_argument(
        "--initialization",
        choices=("random", "imagenet-v1"),
        default="random",
        help="random exactly matches the deleted launcher; imagenet-v1 tests its inert pretrained_bn intent",
    )
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    cfg = get_dataset(args.dataset_name)
    args.dataset_name = cfg.name
    args.dataset_dir = str(args.data_dir.resolve())
    args.mean_norm = list(cfg.mean)
    args.std_norm = list(cfg.std)
    args.ncls = cfg.classes
    args.input_size = 224
    args.use_multi_gpu = False
    args.base_seed = args.seed

    args.output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = args.output_dir / "complete.json"
    if args.skip_completed and completion_path.is_file():
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if (completion.get("status") == "complete" and
                completion.get("seed") == args.seed and
                completion.get("initialization", "random") == args.initialization):
            print(f"Historical plain teacher already complete: {completion_path}")
            return

    seed_everything(args.seed)
    train_loader, test_loader = load_dataset(0, args)
    use_imagenet_weights = args.initialization == "imagenet-v1"
    model = load_model(
        "ResNet18", cfg.classes, "torchvision", use_imagenet_weights, True
    ).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.eta_min
    )

    manifest = {
        "status": "running",
        "dataset": cfg.name,
        "seed": args.seed,
        "initialization": args.initialization,
        "source_commit": SOURCE_COMMIT,
        "source_trainer_blob": SOURCE_TRAINER_BLOB,
        "source_launcher_blob": SOURCE_LAUNCHER_BLOB,
        "historical_launcher": "squeeze/scripts/squeeze_CUB_imsize224_resnet18.sh",
        "initial_weights": (
            "torchvision ResNet18 IMAGENET1K_V1"
            if use_imagenet_weights else "random torchvision ResNet18"
        ),
        "pretrained_bn_flag": True,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": "SGD",
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR",
        "eta_min": args.eta_min,
        "augmentation": "HorizontalFlip+ColorJitter(0.2,0.2,0.2,0.1)+Rotation(15deg)",
        "checkpoint_selection": "final epoch, matching deleted upstream trainer",
        "provenance_extension": (
            "Explicit global seed and manifests; numerical launcher values are unchanged."
            if not use_imagenet_weights else
            "Single-variable diagnostic: enable ImageNet-V1 weights suggested by the otherwise inert pretrained_bn flag."
        ),
    }
    atomic_json_dump(manifest, args.output_dir / "manifest.json")
    metrics_path = args.output_dir / "metrics.jsonl"
    best_accuracy = -1.0
    best_epoch = -1
    final_accuracy = None
    final_loss = None

    for epoch in range(args.epochs):
        started = time.time()
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            train_accuracy, train_loss = evaluate_loader(
                model, criterion, train_loader, "cuda"
            )
            validation_accuracy, validation_loss = evaluate_loader(
                model, criterion, test_loader, "cuda"
            )
            train_accuracy = 100.0 * float(train_accuracy)
            validation_accuracy = 100.0 * float(validation_accuracy)
            record = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_accuracy": train_accuracy,
                "train_loss": float(train_loss),
                "validation_accuracy": validation_accuracy,
                "validation_loss": float(validation_loss),
                "seconds": time.time() - started,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                best_epoch = epoch
            if epoch == args.epochs - 1:
                final_accuracy = validation_accuracy
                final_loss = float(validation_loss)
        scheduler.step()

    if final_accuracy is None:
        raise RuntimeError("Final checkpoint was not evaluated")
    atomic_torch_save(model.state_dict(), args.output_dir / "ResNet18.pth")
    completion = {
        **manifest,
        "status": "complete",
        "best_validation_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "selected_validation_accuracy": final_accuracy,
        "selected_epoch": args.epochs - 1,
        "final_validation_loss": final_loss,
    }
    atomic_json_dump(completion, completion_path)
    print(json.dumps(completion, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
