"""Recover FD2 images on the standard SRe2L++ foundation.

The implementation is deliberately independent of ``recover.py`` so FD2 does
not introduce conditional behavior into the audited plain baseline.
"""

from __future__ import annotations

import argparse
import collections
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
from PIL import Image
from torchvision import models, transforms
from torchvision.io import read_image

FINE_GRAINED_DIR = Path(__file__).resolve().parents[1] / "fine_grained"
sys.path.insert(0, str(FINE_GRAINED_DIR))
from config import get_dataset  # noqa: E402
from fd2_standard_components import (  # noqa: E402
    BNFeatureHook,
    CAL,
    LastFeatureHook,
    compose_recovery_loss,
    dataset_setting,
    fine_grained_characteristic_loss,
    previous_ids_in_group,
    semantic_setting,
    similarity_loss,
)


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_lr(optimizer, base_lr: float, iteration: int, iterations: int) -> None:
    lr = 0.5 * (1.0 + np.cos(np.pi * iteration / iterations)) * base_lr
    for group in optimizer.param_groups:
        group["lr"] = lr


def initialize_patch_data(args, targets: torch.Tensor, ipc_id: int, device) -> torch.Tensor:
    images = []
    for target in targets.tolist():
        class_name = f"{target:05d}"
        path = args.patch_dir / "2" / class_name / f"class{class_name}_id{ipc_id:05d}.jpg"
        if not path.is_file():
            raise FileNotFoundError(f"Missing standard 2x2 initialization patch: {path}")
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        array = (array - np.asarray(args.mean, dtype=np.float32)) / np.asarray(args.std, dtype=np.float32)
        images.append(torch.from_numpy(array.transpose(2, 0, 1)))
    return torch.stack(images).to(device).requires_grad_(True)


def clip_normalized(images: torch.Tensor, mean, std) -> torch.Tensor:
    for channel, (channel_mean, channel_std) in enumerate(zip(mean, std)):
        images[:, channel].clamp_(-channel_mean / channel_std, (1.0 - channel_mean) / channel_std)
    return images


def load_teacher(path: Path, classes: int, attention_maps: int, semantics: str, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("semantics") != semantics:
        raise RuntimeError(
            f"Teacher semantics {checkpoint.get('semantics')!r} do not match requested {semantics!r}"
        )
    if checkpoint.get("attention_maps") != attention_maps:
        raise RuntimeError("Teacher attention-map count does not match dataset setting")
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Linear(backbone.fc.in_features, classes)
    backbone.load_state_dict(checkpoint["ResNet18"])
    cal = CAL(classes, attention_maps)
    cal.load_state_dict(checkpoint["cal"])
    backbone.to(device).eval()
    cal.to(device).eval()
    for parameter in list(backbone.parameters()) + list(cal.parameters()):
        parameter.requires_grad_(False)
    return backbone, cal, checkpoint["feature_center"].to(device)


def bn_hooks(module: nn.Module):
    return [BNFeatureHook(item) for item in module.modules() if isinstance(item, nn.BatchNorm2d)]


def bn_loss(hooks, first_multiplier: float = 1.0):
    if not hooks:
        # The caller always adds this to GPU losses.
        return torch.tensor(0.0, device="cuda")
    weights = [first_multiplier] + [1.0] * (len(hooks) - 1)
    return sum(hook.r_feature * weight for hook, weight in zip(hooks, weights))


def load_previous_attentions(
    args,
    targets: torch.Tensor,
    previous_ids: tuple[int, ...],
    backbone,
    cal,
    feature_hook,
    device,
):
    if not previous_ids:
        return []
    normalize = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(args.mean, args.std),
        ]
    )
    released_augmentation = transforms.Compose(
        [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
    )
    result = []
    with torch.no_grad():
        for target in targets.tolist():
            images = []
            for ipc_id in previous_ids:
                path = args.output_dir / f"new{target:03d}" / f"class{target:03d}_id{ipc_id:03d}.jpg"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing preceding group image: {path}")
                image = normalize(read_image(str(path)))
                if args.semantics == "released_semantics":
                    image = released_augmentation(image)
                images.append(image)
            backbone(torch.stack(images).to(device))
            _raw, _effect, _features, _attention, maps = cal(feature_hook.feature)
            result.append(maps.detach())
    return result


def save_images(inputs, targets, ipc_id: int, output_dir: Path, mean, std):
    mean_tensor = torch.tensor(mean, device=inputs.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=inputs.device).view(1, 3, 1, 1)
    images = (inputs.detach() * std_tensor + mean_tensor).clamp(0, 1).cpu()
    for image, target in zip(images, targets.cpu()):
        class_id = int(target.item())
        class_dir = output_dir / f"new{class_id:03d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        array = image.numpy().transpose(1, 2, 0)
        Image.fromarray((array * 255).astype(np.uint8)).save(
            class_dir / f"class{class_id:03d}_id{ipc_id:03d}.jpg"
        )


def batch_is_complete(output_dir: Path, targets: torch.Tensor, ipc_id: int) -> bool:
    return all(
        (output_dir / f"new{int(target):03d}" / f"class{int(target):03d}_id{ipc_id:03d}.jpg").is_file()
        for target in targets.tolist()
    )


def recover_batch(args, ipc_id, targets, backbone, cal, centers, feature_hook, backbone_bn, cal_bn, device):
    if args.skip_completed and batch_is_complete(args.output_dir, targets, ipc_id):
        print(f"ipc={ipc_id} labels={targets[0]}..{targets[-1]} already complete", flush=True)
        return
    targets = targets.to(device)
    previous_ids = previous_ids_in_group(ipc_id, args.group_size)
    previous_attention = load_previous_attentions(
        args, targets.cpu(), previous_ids, backbone, cal, feature_hook, device
    )
    # The standard patch contract maps global IPC id directly to patch id.
    inputs = initialize_patch_data(args, targets.cpu(), ipc_id, device)
    optimizer = torch.optim.Adam([inputs], lr=args.lr, betas=(0.5, 0.9), eps=1e-8)
    criterion = nn.CrossEntropyLoss().to(device)
    augmentation = transforms.Compose(
        [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
    )
    started = time.time()
    for iteration in range(args.iterations):
        cosine_lr(optimizer, args.lr, iteration, args.iterations)
        current = augmentation(inputs) if args.apply_data_augmentation else inputs
        current = torch.roll(
            current,
            shifts=(random.randint(0, args.jitter), random.randint(0, args.jitter)),
            dims=(2, 3),
        )
        optimizer.zero_grad(set_to_none=True)
        backbone_logits = backbone(current)
        raw, _effect, features, _attention, attention_maps = cal(feature_hook.feature)
        ce_backbone = criterion(backbone_logits, targets)
        ce_cal = criterion(raw, targets)
        feature = fine_grained_characteristic_loss(
            features, targets, centers, beta=args.intra_feature_weight
        )
        if previous_ids:
            sample_losses = [
                similarity_loss(attention_maps[index : index + 1], previous_attention[index])
                for index in range(targets.numel())
            ]
            similarity = torch.stack(sample_losses).mean()
        else:
            similarity = torch.zeros((), device=device)
        backbone_bn_value = bn_loss(backbone_bn, args.first_bn_multiplier)
        cal_bn_value = bn_loss(cal_bn) if cal_bn else torch.zeros((), device=device)
        loss, parts = compose_recovery_loss(
            args.semantics,
            ce_backbone=ce_backbone,
            ce_cal=ce_cal,
            bn_backbone=backbone_bn_value,
            bn_cal=cal_bn_value,
            feature_loss=feature,
            similarity=similarity,
            cal_ratio=args.cal_ratio,
            r_bn=args.r_bn,
        )
        loss.backward()
        optimizer.step()
        inputs.data = clip_normalized(inputs.data, args.mean, args.std)
        if iteration % 100 == 0 or iteration == args.iterations - 1:
            printable = {key: float(value.detach()) for key, value in parts.items()}
            printable.update(
                ipc_id=ipc_id,
                iteration=iteration,
                labels=[int(targets[0]), int(targets[-1])],
                total=float(loss.detach()),
                seconds=time.time() - started,
            )
            print(json.dumps(printable, sort_keys=True), flush=True)
            started = time.time()
    save_images(inputs, targets, ipc_id, args.output_dir, args.mean, args.std)
    optimizer.state = collections.defaultdict(dict)
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser("Standard-protocol FD2 recovery")
    parser.add_argument("--semantics", required=True, choices=("released_semantics", "paper_literal"))
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--patch-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--r-bn", type=float, default=1e-3)
    parser.add_argument("--first-bn-multiplier", type=float, default=10.0)
    parser.add_argument("--jitter", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--intra-feature-weight", type=float, default=0.5)
    parser.add_argument("--ipc-start", type=int, default=0)
    parser.add_argument("--ipc-end", type=int, default=5)
    parser.add_argument("--apply-data-augmentation", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    if args.group_size != 4:
        raise ValueError("The frozen FD2 protocol requires N_S=4")
    cfg = get_dataset(args.dataset_name)
    fd2_cfg = dataset_setting(cfg.name)
    setting = semantic_setting(args.semantics)
    args.mean = list(cfg.mean)
    args.std = list(cfg.std)
    args.cal_ratio = fd2_cfg.cal_ratio
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patch_manifest = args.patch_dir / "2" / "patch_manifest.json"
    if not patch_manifest.is_file():
        raise FileNotFoundError(f"Missing standard patch manifest: {patch_manifest}")
    seed_everything(args.seed)
    device = torch.device("cuda")
    backbone, cal, centers = load_teacher(
        args.teacher, cfg.classes, fd2_cfg.attention_maps, args.semantics, device
    )
    feature_hook = LastFeatureHook(backbone.layer4[1].bn2)
    backbone_bn = bn_hooks(backbone)
    cal_bn = bn_hooks(cal) if setting.recovery_cal_bn else []
    manifest = {
        "status": "running",
        "method": "FD2 on standard SRe2L++ foundation",
        "semantics": args.semantics,
        "semantic_setting": setting.to_dict(),
        "paper_reference": "arXiv:2603.25144v2 (2026-06-27)",
        "released_reference": [
            "FD2/recover/recover_FD2.py",
            "FD2/recover/utils_recover.py",
        ],
        "dataset": cfg.name,
        "seed": args.seed,
        "teacher": str(args.teacher.resolve()),
        "teacher_sha256": sha256(args.teacher),
        "patch_dir": str(args.patch_dir.resolve()),
        "patch_manifest": str(patch_manifest.resolve()),
        "patch_manifest_sha256": sha256(patch_manifest),
        "patch_mapping": "global ipc_id == patch id",
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "optimizer": "Adam",
        "lr": args.lr,
        "betas": [0.5, 0.9],
        "r_bn": args.r_bn,
        "first_bn_multiplier": args.first_bn_multiplier,
        "jitter": args.jitter,
        "group_size": args.group_size,
        "ipc_range": [args.ipc_start, args.ipc_end],
        "attention_maps": fd2_cfg.attention_maps,
        "cal_ratio": fd2_cfg.cal_ratio,
    }
    manifest_path = args.output_dir.parent / "recovery_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        protected = (
            "semantics",
            "dataset",
            "seed",
            "teacher_sha256",
            "patch_manifest_sha256",
            "iterations",
            "batch_size",
            "lr",
            "r_bn",
            "first_bn_multiplier",
            "jitter",
            "group_size",
            "ipc_range",
            "attention_maps",
            "cal_ratio",
        )
        mismatches = [key for key in protected if existing.get(key) != manifest.get(key)]
        if mismatches and next(args.output_dir.glob("new*/class*_id*.jpg"), None) is not None:
            raise RuntimeError(
                f"Refusing to reuse recovery images under incompatible settings {mismatches}: {manifest_path}"
            )
    atomic_json(manifest, manifest_path)
    targets = torch.arange(cfg.classes, dtype=torch.long)
    for ipc_id in range(args.ipc_start, args.ipc_end):
        for start in range(0, cfg.classes, args.batch_size):
            recover_batch(
                args,
                ipc_id,
                targets[start : start + args.batch_size],
                backbone,
                cal,
                centers,
                feature_hook,
                backbone_bn,
                cal_bn,
                device,
            )
    expected = cfg.classes * (args.ipc_end - args.ipc_start)
    actual = len(list(args.output_dir.glob("new*/class*_id*.jpg")))
    if actual != expected:
        raise RuntimeError(f"Recovery output count {actual} != expected {expected}")
    atomic_json({**manifest, "status": "complete", "image_count": actual}, manifest_path)
    feature_hook.close()
    for hook in backbone_bn + cal_bn:
        hook.close()


if __name__ == "__main__":
    main()
