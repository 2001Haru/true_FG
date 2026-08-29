import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from config import get_dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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


def class_images(train_dir: Path) -> list[tuple[str, list[Path]]]:
    result = []
    for class_dir in sorted(path for path in train_dir.iterdir() if path.is_dir()):
        images = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        result.append((class_dir.name, images))
    return result


def center_pad(images: torch.Tensor, output_size: int) -> torch.Tensor:
    vertical = output_size - images.shape[-2]
    horizontal = output_size - images.shape[-1]
    return F.pad(
        images,
        (
            horizontal // 2,
            horizontal - horizontal // 2,
            vertical // 2,
            vertical - vertical // 2,
        ),
    )


def make_mosaic(parts: torch.Tensor, output_size: int) -> torch.Tensor:
    if parts.shape[0] != 4:
        raise ValueError(f"2x2 initialization requires four parts, got {parts.shape[0]}")
    half = output_size // 2
    mosaic = torch.empty(3, output_size, output_size, dtype=parts.dtype)
    # Match FD2/RDED_patch.py, whose F.interpolate call uses the default
    # nearest-neighbor mode when assembling the 2x2 initialization.
    mosaic[:, :half, :half] = F.interpolate(parts[0:1], (half, half))[0]
    mosaic[:, :half, half:] = F.interpolate(parts[1:2], (half, output_size - half))[0]
    mosaic[:, half:, :half] = F.interpolate(parts[2:3], (output_size - half, half))[0]
    mosaic[:, half:, half:] = F.interpolate(parts[3:4], (output_size - half, output_size - half))[0]
    return mosaic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Create FD2-style 2x2 SRe2L++ initialization patches")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Difficulty directory consumed as --patch-diff 2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patches-per-class", type=int, default=5)
    parser.add_argument("--num-crops", type=int, default=5)
    parser.add_argument("--candidate-count", type=int, default=0,
                        help="0 uses the minimum training samples per class")
    parser.add_argument("--forward-batch-size", type=int, default=0,
                        help="0 uses candidate-count, matching the released CUB utility")
    parser.add_argument("--torch-cpu-threads", type=int, default=1,
                        help="intra-op threads for small CPU crop/resize tensors")
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = get_dataset(args.dataset_name)
    if args.torch_cpu_threads < 1:
        raise ValueError("--torch-cpu-threads must be positive")
    # RandomResizedCrop operates on one small tensor at a time. Letting each
    # interpolation fan out over every CPU core makes thread-launch overhead
    # dominate and starves parallel dataset jobs; it does not change crop RNG.
    torch.set_num_threads(args.torch_cpu_threads)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for patch scoring")
    if not args.teacher.is_file():
        raise FileNotFoundError(args.teacher)
    train_dir = args.data_dir / "train"
    classes = class_images(train_dir)
    if len(classes) != cfg.classes:
        raise RuntimeError(f"Found {len(classes)} classes, expected {cfg.classes}")
    minimum = min(len(images) for _, images in classes)
    candidate_count = args.candidate_count or minimum
    if candidate_count > minimum or candidate_count < 4:
        raise ValueError(f"candidate-count={candidate_count}, allowed range is 4..{minimum}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = args.output_dir / "patch_manifest.json"
    expected_files = cfg.classes * args.patches_per_class
    existing_files = list(args.output_dir.glob("*/*.jpg"))
    if args.skip_completed and completion_path.is_file() and len(existing_files) == expected_files:
        print(f"Patches already complete: {args.output_dir}", flush=True)
        return

    state_dict = torch.load(args.teacher, map_location="cpu", weights_only=True)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, cfg.classes)
    model.load_state_dict(state_dict, strict=True)
    model.cuda().eval()

    crop = transforms.RandomResizedCrop(
        224 // 2,
        ratio=(1.0, 1.0),
        antialias=True,
    )
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(cfg.mean, cfg.std)
    mean = torch.tensor(cfg.mean).view(3, 1, 1)
    std = torch.tensor(cfg.std).view(3, 1, 1)
    forward_batch_size = args.forward_batch_size or candidate_count

    for patch_id in range(args.patches_per_class):
        for class_id, (class_name, paths) in enumerate(classes):
            output_class = args.output_dir / f"{class_id:05d}"
            output_class.mkdir(parents=True, exist_ok=True)
            output_path = output_class / f"class{class_id:05d}_id{patch_id:05d}.jpg"
            if args.skip_completed and output_path.is_file():
                continue

            group_seed = args.seed * 1_000_003 + patch_id * cfg.classes + class_id
            random.seed(group_seed)
            np.random.seed(group_seed % (2**32))
            torch.manual_seed(group_seed)
            candidates = []
            for path in paths[:candidate_count]:
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    tensor = to_tensor(image)
                candidates.append(torch.stack([normalize(crop(tensor)) for _ in range(args.num_crops)]))
            images = torch.stack(candidates)  # [candidate, crop, C, 112, 112]
            flattened = images.permute(1, 0, 2, 3, 4).reshape(
                args.num_crops * candidate_count, 3, 112, 112
            )
            labels = torch.full(
                (args.num_crops * candidate_count,), class_id, dtype=torch.long, device="cuda"
            )
            logits = []
            padded = center_pad(flattened, 224)
            for start in range(0, padded.shape[0], forward_batch_size):
                logits.append(model(padded[start:start + forward_batch_size].cuda(non_blocking=True)))
            losses = F.cross_entropy(torch.cat(logits), labels, reduction="none")
            losses = losses.reshape(args.num_crops, candidate_count)
            best_crop = losses.argmin(0)
            candidate_index = torch.arange(candidate_count, device="cuda")
            best_losses = losses[best_crop, candidate_index]
            best_images = flattened.reshape(
                args.num_crops, candidate_count, 3, 112, 112
            )[best_crop.cpu(), torch.arange(candidate_count)]
            selected = best_losses.argsort()[:4].cpu()
            selected_images = best_images[selected]
            denormalized = (selected_images * std + mean).clamp(0, 1)
            mosaic = make_mosaic(denormalized, 224)
            array = (mosaic.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(array).save(output_path)
            print(
                f"patch={patch_id} class={class_id}/{cfg.classes - 1} "
                f"name={class_name} ce={best_losses[selected].tolist()}",
                flush=True,
            )

    files = list(args.output_dir.glob("*/*.jpg"))
    if len(files) != expected_files:
        raise RuntimeError(f"Generated {len(files)} patch files, expected {expected_files}")
    atomic_json_dump(
        {
            "status": "complete",
            "dataset": cfg.to_dict(),
            "data_dir": str(args.data_dir.resolve()),
            "teacher": str(args.teacher.resolve()),
            "teacher_sha256": sha256(args.teacher),
            "seed": args.seed,
            "candidate_count": candidate_count,
            "minimum_class_count": minimum,
            "patches_per_class": args.patches_per_class,
            "num_crops": args.num_crops,
            "factor": 2,
            "files": len(files),
        },
        completion_path,
    )
    print(f"Patch generation complete: {len(files)} files at {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
