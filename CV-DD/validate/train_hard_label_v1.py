"""Update-budgeted ImageNet-initialized evaluator for hard-label FG images."""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchvision.transforms import InterpolationMode


CURRENT_DIR = Path(__file__).resolve().parent
FINE_GRAINED_DIR = CURRENT_DIR.parent / "fine_grained"
PROTOCOL_SPEC_PATH = FINE_GRAINED_DIR / "hard_label_v1_protocol.json"
sys.path.insert(0, str(FINE_GRAINED_DIR))
from hard_label_v1_protocol import (  # noqa: E402
    BACKBONE_LR,
    BACKBONE_MIN_LR,
    EVAL_EVERY_UPDATES,
    GRADIENT_ACCUMULATION_STEPS,
    HEAD_LR,
    HEAD_MIN_LR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGENET_V1_FILE_SHA256,
    MOMENTUM,
    PHYSICAL_BATCH_SIZE,
    PROTOCOL_NAME,
    RESIZE_SIZE,
    TOTAL_UPDATES,
    TRAIN_WORKERS,
    VALIDATION_BATCH_SIZE,
    WEIGHT_DECAY,
    cosine_lr,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    train_source = parser.add_mutually_exclusive_group(required=True)
    train_source.add_argument("--train-dir", type=Path)
    train_source.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-dir", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--num-classes", required=True, type=int)
    parser.add_argument("--ipc", required=True, type=int)
    parser.add_argument("--student-seed", required=True, type=int)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--imagenet-weights-path", required=True, type=Path)
    parser.add_argument("--total-updates", default=TOTAL_UPDATES, type=int)
    parser.add_argument("--batch-size", default=PHYSICAL_BATCH_SIZE, type=int)
    parser.add_argument("--backbone-lr", default=BACKBONE_LR, type=float)
    parser.add_argument("--head-lr", default=HEAD_LR, type=float)
    parser.add_argument("--backbone-min-lr", default=BACKBONE_MIN_LR, type=float)
    parser.add_argument("--head-min-lr", default=HEAD_MIN_LR, type=float)
    parser.add_argument("--momentum", default=MOMENTUM, type=float)
    parser.add_argument("--weight-decay", default=WEIGHT_DECAY, type=float)
    parser.add_argument("--eval-every-updates", default=EVAL_EVERY_UPDATES, type=int)
    parser.add_argument("--workers", default=TRAIN_WORKERS, type=int)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--val-batch-size", default=VALIDATION_BATCH_SIZE, type=int)
    args = parser.parse_args()
    if args.total_updates <= 0 or args.batch_size <= 0 or args.eval_every_updates <= 0:
        parser.error("update, batch, and evaluation intervals must be positive")
    if args.total_updates % args.eval_every_updates != 0:
        parser.error("total updates must be divisible by eval-every-updates")
    if args.num_classes <= 0 or args.ipc <= 0:
        parser.error("num-classes and ipc must be positive")
    if not 0 <= args.backbone_min_lr <= args.backbone_lr:
        parser.error("backbone LR must satisfy 0 <= minimum <= initial")
    if not 0 <= args.head_min_lr <= args.head_lr:
        parser.error("head LR must satisfy 0 <= minimum <= initial")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_transform = transforms.Compose(
        [
            transforms.Resize(RESIZE_SIZE, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(RESIZE_SIZE, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, validation_transform


def split_parameter_groups(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> tuple[list[dict], dict[str, list[str]]]:
    buckets = {
        "backbone_decay": [],
        "backbone_no_decay": [],
        "head_decay": [],
        "head_no_decay": [],
    }
    names = {key: [] for key in buckets}
    for name, parameter in model.named_parameters():
        is_head = name.startswith("fc.")
        use_decay = parameter.ndim > 1 and name.endswith("weight")
        key = f"{'head' if is_head else 'backbone'}_{'decay' if use_decay else 'no_decay'}"
        buckets[key].append(parameter)
        names[key].append(name)
    groups = [
        {
            "params": buckets["backbone_decay"],
            "lr": backbone_lr,
            "weight_decay": weight_decay,
            "group_name": "backbone_decay",
        },
        {
            "params": buckets["backbone_no_decay"],
            "lr": backbone_lr,
            "weight_decay": 0.0,
            "group_name": "backbone_no_decay",
        },
        {
            "params": buckets["head_decay"],
            "lr": head_lr,
            "weight_decay": weight_decay,
            "group_name": "head_decay",
        },
        {
            "params": buckets["head_no_decay"],
            "lr": head_lr,
            "weight_decay": 0.0,
            "group_name": "head_no_decay",
        },
    ]
    if any(not group["params"] for group in groups):
        raise RuntimeError("one or more optimizer parameter groups are empty")
    flattened = [parameter for group in groups for parameter in group["params"]]
    if len(flattened) != len(list(model.parameters())) or len({id(p) for p in flattened}) != len(flattened):
        raise RuntimeError("optimizer parameter groups do not partition model parameters")
    return groups, names


def set_learning_rates(optimizer: torch.optim.Optimizer, args: argparse.Namespace, t: int) -> None:
    backbone_lr = cosine_lr(args.backbone_lr, args.backbone_min_lr, t, args.total_updates)
    head_lr = cosine_lr(args.head_lr, args.head_min_lr, t, args.total_updates)
    for group in optimizer.param_groups:
        group["lr"] = head_lr if group["group_name"].startswith("head") else backbone_lr


@torch.no_grad()
def validate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> dict:
    model.eval()
    loss_sum = 0.0
    correct1 = 0
    correct5 = 0
    count = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += criterion(logits, targets).item()
        predictions = logits.topk(5, dim=1).indices
        correct1 += predictions[:, :1].eq(targets[:, None]).sum().item()
        correct5 += predictions.eq(targets[:, None]).sum().item()
        count += targets.numel()
    return {
        "loss": loss_sum / count,
        "top1": 100.0 * correct1 / count,
        "top5": 100.0 * correct5 / count,
        "images": count,
    }


@torch.no_grad()
def per_class_accuracy(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    classes: list[str],
) -> list[dict]:
    model.eval()
    correct = torch.zeros(len(classes), dtype=torch.long)
    total = torch.zeros(len(classes), dtype=torch.long)
    for images, targets in loader:
        predictions = model(images.to(device, non_blocking=True)).argmax(1).cpu()
        targets = targets.long()
        total.scatter_add_(0, targets, torch.ones_like(targets))
        correct.scatter_add_(0, targets, predictions.eq(targets).long())
    return [
        {
            "class_id": index,
            "class_name": classes[index],
            "correct": int(correct[index]),
            "total": int(total[index]),
            "accuracy": 100.0 * int(correct[index]) / max(int(total[index]), 1),
        }
        for index in range(len(classes))
    ]


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class ManifestImageDataset(torch.utils.data.Dataset):
    """ImageFolder-equivalent source list backed by an audited selection manifest."""

    def __init__(self, manifest_path: Path, transform) -> None:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise RuntimeError(f"selection manifest is incomplete: {manifest_path}")
        rows = sorted(
            payload["images"],
            key=lambda row: (row["class_folder"], row["source_path"]),
        )
        self.classes = sorted({row["class_folder"] for row in rows})
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.samples = []
        for row in rows:
            target = self.class_to_idx[row["class_folder"]]
            if int(row["class_id"]) != target:
                raise RuntimeError(
                    f"manifest class mapping mismatch: {row['source_path']}"
                )
            source = Path(row["source_path"])
            if not source.is_file():
                raise FileNotFoundError(source)
            self.samples.append((str(source), target))
        if len({path for path, _ in self.samples}) != len(self.samples):
            raise RuntimeError("selection manifest contains duplicate source paths")
        self.targets = [target for _, target in self.samples]
        self.transform = transform
        self.manifest_path = manifest_path.resolve()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = datasets.folder.default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def main() -> None:
    args = parse_args()
    seed_everything(args.student_seed)
    device = torch.device("cuda")
    train_transform, validation_transform = make_transforms()
    validation_dataset = datasets.ImageFolder(args.val_dir, transform=validation_transform)
    if args.train_manifest is not None:
        train_dataset = ManifestImageDataset(args.train_manifest, transform=train_transform)
        train_source_type = "selection_manifest"
        train_source = str(args.train_manifest.resolve())
    else:
        train_dataset = datasets.ImageFolder(args.train_dir, transform=train_transform)
        train_source_type = "imagefolder"
        train_source = str(args.train_dir.resolve())
    if train_dataset.classes != validation_dataset.classes:
        raise RuntimeError("train and validation class folders differ")
    if len(train_dataset.classes) != args.num_classes:
        raise RuntimeError(
            f"class count {len(train_dataset.classes)} != expected {args.num_classes}"
        )
    expected_images = args.num_classes * args.ipc
    if len(train_dataset) != expected_images:
        raise RuntimeError(f"train images {len(train_dataset)} != classes*IPC {expected_images}")

    loader_generator = torch.Generator().manual_seed(args.student_seed)
    persistent = args.persistent_workers and args.workers > 0
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=args.workers,
        persistent_workers=persistent,
        pin_memory=True,
        prefetch_factor=2 if args.workers > 0 else None,
        drop_last=False,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=persistent,
        pin_memory=True,
        prefetch_factor=2 if args.workers > 0 else None,
        drop_last=False,
    )

    weights = models.ResNet18_Weights.IMAGENET1K_V1
    if not args.imagenet_weights_path.is_file():
        raise FileNotFoundError(args.imagenet_weights_path)
    weights_file_hash = file_sha256(args.imagenet_weights_path)
    if weights_file_hash != IMAGENET_V1_FILE_SHA256:
        raise RuntimeError(
            f"ImageNet-V1 weights SHA-256 {weights_file_hash} != {IMAGENET_V1_FILE_SHA256}"
        )
    model = models.resnet18(weights=None)
    model.load_state_dict(
        torch.load(args.imagenet_weights_path, map_location="cpu", weights_only=True),
        strict=True,
    )
    imagenet_state_sha256 = state_dict_sha256(model.state_dict())
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    initial_head_sha256 = state_dict_sha256(model.fc.state_dict())
    optimizer_groups, parameter_names = split_parameter_groups(
        model, args.backbone_lr, args.head_lr, args.weight_decay
    )
    optimizer = torch.optim.SGD(
        optimizer_groups,
        lr=args.backbone_lr,
        momentum=args.momentum,
        dampening=0.0,
        nesterov=False,
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = []
    updates_completed = 0
    epochs_started = 0
    examples_seen = 0
    best_top1 = float("-inf")
    best_update = None
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    while updates_completed < args.total_updates:
        epochs_started += 1
        model.train()
        for images, targets in train_loader:
            if updates_completed >= args.total_updates:
                break
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            backbone_lr_used = optimizer.param_groups[0]["lr"]
            head_lr_used = optimizer.param_groups[2]["lr"]
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            updates_completed += 1
            examples_seen += targets.numel()
            set_learning_rates(optimizer, args, updates_completed)

            if updates_completed % 50 == 0 or updates_completed == 1:
                print(
                    f"update={updates_completed}/{args.total_updates} "
                    f"loss={loss.item():.6f} backbone_lr={backbone_lr_used:.9g} "
                    f"head_lr={head_lr_used:.9g}",
                    flush=True,
                )
            if updates_completed % args.eval_every_updates == 0:
                metrics = validate(model, validation_loader, device)
                metrics.update(
                    update=updates_completed,
                    backbone_lr_used=backbone_lr_used,
                    head_lr_used=head_lr_used,
                )
                history.append(metrics)
                if metrics["top1"] > best_top1:
                    best_top1 = metrics["top1"]
                    best_update = updates_completed
                print(json.dumps({"validation": metrics}, sort_keys=True), flush=True)
                if updates_completed < args.total_updates:
                    model.train()

    final_metrics = history[-1]
    per_class = per_class_accuracy(model, validation_loader, device, validation_dataset.classes)
    final_checkpoint = args.checkpoint_dir / "final.pth.tar"
    torch.save(
        {
            "protocol": PROTOCOL_NAME,
            "updates_completed": updates_completed,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "student_seed": args.student_seed,
        },
        final_checkpoint,
    )
    payload = {
        "status": "complete",
        "protocol": PROTOCOL_NAME,
        "protocol_spec": str(PROTOCOL_SPEC_PATH.resolve()),
        "protocol_spec_sha256": file_sha256(PROTOCOL_SPEC_PATH),
        "dataset": args.dataset_name,
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
        "imagenet_weights_enum": "ResNet18_Weights.IMAGENET1K_V1",
        "imagenet_weights_url": weights.url,
        "imagenet_weights_file": str(args.imagenet_weights_path.resolve()),
        "imagenet_weights_file_sha256": weights_file_hash,
        "imagenet_initial_state_sha256": imagenet_state_sha256,
        "initial_head_sha256": initial_head_sha256,
        "student_seed": args.student_seed,
        "optimizer": "sgd",
        "momentum": args.momentum,
        "dampening": 0.0,
        "nesterov": False,
        "weight_decay": args.weight_decay,
        "weight_decay_excludes": "all_1d_parameters_and_all_biases_including_bn",
        "optimizer_parameter_names": parameter_names,
        "backbone_initial_lr": args.backbone_lr,
        "head_initial_lr": args.head_lr,
        "backbone_min_lr": args.backbone_min_lr,
        "head_min_lr": args.head_min_lr,
        "scheduler": "per_update_full_cosine",
        "scheduler_formula": "eta_min + (eta_0-eta_min)/2 * (1+cos(pi*t/U))",
        "scheduler_step_semantics": "update k uses t=k-1; state reaches t=U after update U",
        "scheduler_endpoint_backbone_lr": optimizer.param_groups[0]["lr"],
        "scheduler_endpoint_head_lr": optimizer.param_groups[2]["lr"],
        "warmup_updates": 0,
        "total_optimizer_updates": args.total_updates,
        "updates_completed": updates_completed,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "physical_batch_size": args.batch_size,
        "drop_last": False,
        "epochs_started": epochs_started,
        "examples_seen": examples_seen,
        "bn_running_statistics": "updated_in_train_mode",
        "normalization_mean": list(IMAGENET_MEAN),
        "normalization_std": list(IMAGENET_STD),
        "train_transform": "Resize(256,bilinear)->RandomCrop(224)->HorizontalFlip(p=0.5)",
        "test_transform": "Resize(256,bilinear)->CenterCrop(224)",
        "image_size": IMAGE_SIZE,
        "train_images": len(train_dataset),
        "validation_images": len(validation_dataset),
        "num_classes": args.num_classes,
        "dataloader_workers": args.workers,
        "persistent_workers": persistent,
        "prefetch_factor": 2 if args.workers > 0 else None,
        "validation_batch_size": args.val_batch_size,
        "evaluation_every_updates": args.eval_every_updates,
        "primary_metric": "final_update_top1",
        "final_top1": final_metrics["top1"],
        "final_top5": final_metrics["top5"],
        "final_loss": final_metrics["loss"],
        "best_top1_diagnostic": best_top1,
        "best_update_diagnostic": best_update,
        "validation_history": history,
        "per_class_final": per_class,
        "final_checkpoint": str(final_checkpoint.resolve()),
        "final_checkpoint_sha256": file_sha256(final_checkpoint),
        "elapsed_seconds": time.time() - started,
        "train_source_type": train_source_type,
        "train_source": train_source,
        "train_dir": (str(args.train_dir.resolve()) if args.train_dir is not None else None),
        "train_manifest": (
            str(args.train_manifest.resolve()) if args.train_manifest is not None else None
        ),
        "validation_dir": str(args.val_dir.resolve()),
    }
    atomic_json(args.result, payload)
    print(json.dumps({"result": str(args.result), "final_top1": final_metrics["top1"]}))


if __name__ == "__main__":
    main()
