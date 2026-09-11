"""Auditable ImageNette execution of the released RDED validation semantics.

The validation split is deliberately the complete ImageNette split rather than
the released repository's val_ipc=50 subset.
"""

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torchvision.transforms import InterpolationMode


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def state_dict_sha256(state_dict):
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ShufflePatches(nn.Module):
    """Exact row/column permutation used by released RDED."""

    def __init__(self, factor=2):
        super().__init__()
        self.factor = factor

    def _shuffle_width(self, image):
        width = image.shape[-1]
        patch_width = width // self.factor
        patches = [image[..., i * patch_width:(i + 1) * patch_width] for i in range(self.factor)]
        random.shuffle(patches)
        return torch.cat(patches, dim=-1)

    def forward(self, image):
        image = self._shuffle_width(image)
        image = image.permute(0, 2, 1)
        image = self._shuffle_width(image)
        return image.permute(0, 2, 1)


def parameter_groups(model):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if "weight" in name and parameter.ndim > 1 else no_decay).append(parameter)
    return [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]


def load_teacher(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def rand_bbox(size, lam):
    width, height = size[2], size[3]
    ratio = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(width * ratio), int(height * ratio)
    cx, cy = np.random.randint(width), np.random.randint(height)
    return (
        int(np.clip(cx - cut_w // 2, 0, width)),
        int(np.clip(cy - cut_h // 2, 0, height)),
        int(np.clip(cx + cut_w // 2, 0, width)),
        int(np.clip(cy + cut_h // 2, 0, height)),
    )


def cutmix(images):
    permutation = torch.randperm(images.size(0), device=images.device)
    bbox = rand_bbox(images.size(), np.random.beta(1.0, 1.0))
    x1, y1, x2, y2 = bbox
    images[:, :, x1:x2, y1:y2] = images[permutation, :, x1:x2, y1:y2]
    return images


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    class_correct = torch.zeros(10, dtype=torch.long)
    class_total = torch.zeros(10, dtype=torch.long)
    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += F.cross_entropy(logits, targets, reduction="sum").item()
        predictions = logits.argmax(dim=1)
        matches = predictions.eq(targets)
        correct += int(matches.sum())
        total += targets.numel()
        for class_id in range(10):
            mask = targets.eq(class_id)
            class_total[class_id] += int(mask.sum())
            class_correct[class_id] += int(matches[mask].sum())
    return {
        "top1": 100.0 * correct / total,
        "nll": loss_sum / total,
        "count": total,
        "per_class_top1": [100.0 * int(class_correct[i]) / int(class_total[i]) for i in range(10)],
    }


def atomic_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--val-dir", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ipc", required=True, choices=(10, 50), type=int)
    parser.add_argument("--student-seed", required=True, type=int)
    parser.add_argument("--image-arm", required=True, choices=("a_original", "d_rded", "random_real"))
    parser.add_argument("--workers", default=4, type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    if result_path.is_file():
        print(f"result exists: {result_path}")
        return

    random.seed(args.student_seed)
    np.random.seed(args.student_seed)
    torch.manual_seed(args.student_seed)
    torch.cuda.manual_seed_all(args.student_seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    student = models.resnet18(weights=None)
    student.fc = nn.Linear(student.fc.in_features, 10)
    initial_hash = state_dict_sha256(student.state_dict())
    student = student.to(device)
    teacher = load_teacher(args.teacher, device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    augmentation = [transforms.ToTensor()]
    if args.image_arm == "d_rded":
        augmentation.append(ShufflePatches(2))
    augmentation.extend([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(MEAN, STD),
    ])
    train_dataset = datasets.ImageFolder(args.train_dir, transform=transforms.Compose(augmentation))
    expected_train = args.ipc * 10
    if len(train_dataset) != expected_train:
        raise RuntimeError(f"expected {expected_train} train images, found {len(train_dataset)}")
    batch_size = 50 if args.ipc == 10 else 100
    generator = torch.Generator().manual_seed(args.student_seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator,
        num_workers=args.workers, pin_memory=True, persistent_workers=False,
    )
    validation_dataset = datasets.ImageFolder(
        args.val_dir,
        transform=transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        ]),
    )
    if len(validation_dataset) != 3925:
        raise RuntimeError(f"full ImageNette validation must have 3925 images, found {len(validation_dataset)}")
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=256, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=False,
    )

    optimizer = torch.optim.AdamW(parameter_groups(student), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: 0.5 * (1.0 + math.cos(math.pi * epoch / 300 / 2)), last_epoch=-1,
    )
    criterion = nn.KLDivLoss(reduction="batchmean")
    history = []
    updates = 0
    for epoch in range(300):
        student.train()
        loss_sum = samples = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # Released cutmix mutates `images` in place; the metric-only forward
            # therefore sees the same mixed tensor and updates Student BN once.
            mixed = cutmix(images)
            # Preserve released execution: this no-grad forward updates Student
            # BN, followed by the gradient-bearing forward on the same mix.
            with torch.no_grad():
                student(images)
                targets = F.softmax(teacher(mixed) / 20.0, dim=1)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(F.log_softmax(student(mixed) / 20.0, dim=1), targets)
            loss.backward()
            optimizer.step()
            updates += 1
            loss_sum += float(loss) * images.size(0)
            samples += images.size(0)
        evaluation = None
        if (epoch % 10 == 9 or epoch == 299) and epoch > 240:
            evaluation = validate(student, validation_loader, device)
            print(f"epoch={epoch + 1} loss={loss_sum/samples:.6f} top1={evaluation['top1']:.4f}", flush=True)
        elif (epoch + 1) % 10 == 0:
            print(f"epoch={epoch + 1} loss={loss_sum/samples:.6f}", flush=True)
        history.append({"epoch": epoch + 1, "lr_used": optimizer.param_groups[0]["lr"], "train_loss": loss_sum / samples, "evaluation": evaluation})
        scheduler.step()

    evaluations = [(item["epoch"], item["evaluation"]) for item in history if item["evaluation"]]
    best_epoch, best = max(evaluations, key=lambda item: item[1]["top1"])
    final = history[-1]["evaluation"]
    torch.save(student.state_dict(), args.output_dir / "student_final.pth")
    payload = {
        "status": "complete", "dataset": "imagenet-nette", "ipc": args.ipc,
        "image_arm": args.image_arm, "student_seed": args.student_seed,
        "student": "torchvision_resnet18_random", "initial_model_sha256": initial_hash,
        "train_dir": str(args.train_dir.resolve()), "train_images": len(train_dataset),
        "validation_dir": str(args.val_dir.resolve()), "validation_images": len(validation_dataset),
        "teacher_checkpoint": str(args.teacher.resolve()), "teacher_sha256": file_sha256(args.teacher),
        "teacher_mode": "eval", "teacher_labels": "online_same_cutmix_view",
        "student_bn_extra_mixed_no_grad_forward": True,
        "shuffle_patches": args.image_arm == "d_rded", "shuffle_patches_factor": 2 if args.image_arm == "d_rded" else None,
        "rrc_scale": [0.5, 1.0], "rrc_size": 224, "cutmix_alpha": 1.0,
        "temperature_teacher": 20.0, "temperature_student": 20.0,
        "loss": "KLDivLoss(batchmean)", "temperature_squared_compensation": False,
        "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay_matrix_weights": 0.01,
        "weight_decay_bias_bn": 0.0, "betas": [0.9, 0.999],
        "epochs": 300, "batch_size": batch_size, "gradient_accumulation_steps": 1,
        "optimizer_updates": updates, "scheduler": "released_half_cosine_eta2_epoch_step",
        "scheduler_eta": 2.0, "learning_rate_after_final_step": optimizer.param_groups[0]["lr"],
        "workers": args.workers, "persistent_workers": False,
        "validation_schedule": "released last-20-percent window: epochs 250,260,270,280,290,300",
        "best_top1": best["top1"], "best_nll": best["nll"], "best_epoch": best_epoch,
        "final_top1": final["top1"], "final_nll": final["nll"], "final_epoch": 300,
        "final_per_class_top1": final["per_class_top1"], "history": history,
    }
    atomic_json(payload, result_path)
    print(json.dumps({key: payload[key] for key in ("best_top1", "best_epoch", "final_top1", "final_nll", "optimizer_updates")}, indent=2))


if __name__ == "__main__":
    main()
