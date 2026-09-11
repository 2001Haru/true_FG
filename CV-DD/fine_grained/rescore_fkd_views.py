"""Replay an existing FKD geometry schedule on new images and rescore Teacher logits."""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data._utils.fetch import _MapDatasetFetcher
from torchvision import models
from torchvision.transforms import InterpolationMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relabel.utils_fkd import (
    ComposeWithCoords,
    ImageFolder_FKD_MIX,
    RandomHorizontalFlipWithRes,
    RandomResizedCropWithCoords,
    mix_aug,
)


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
_ORIGINAL_FETCH = _MapDatasetFetcher.fetch


def patched_fetch(fetcher, possibly_batched_index):
    if not (hasattr(fetcher.dataset, "mode") and fetcher.dataset.mode == "fkd_load"):
        return _ORIGINAL_FETCH(fetcher, possibly_batched_index)
    mix_index, mix_lam, mix_bbox, soft_label = fetcher.dataset.load_batch_config(
        possibly_batched_index[0]
    )
    if fetcher.auto_collation:
        data = [fetcher.dataset[index] for index in possibly_batched_index]
    else:
        data = fetcher.dataset[possibly_batched_index]
    return fetcher.collate_fn(data), mix_index.cpu(), mix_lam, mix_bbox, soft_label.cpu()


_MapDatasetFetcher.fetch = patched_fetch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_teacher(path: Path) -> nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(state, strict=True)
    return model.cuda().train()


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--source-fkd", required=True, type=Path)
    parser.add_argument("--output-fkd", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--ipc", required=True, type=int, choices=(10, 50))
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--batch-size", default=10, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--fkd-seed", default=42, type=int)
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    manifest_path = args.output_fkd / "relabel_manifest.json"
    expected_batches = args.epochs * args.ipc
    if args.skip_completed and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = list(args.output_fkd.glob("epoch_*/batch_*.tar"))
        if manifest.get("status") == "complete" and len(files) == expected_batches:
            print(f"Rescored FKD already complete: {manifest_path}")
            return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    teacher = load_teacher(args.teacher)
    transform = ComposeWithCoords(
        transforms=[
            RandomResizedCropWithCoords(
                size=224, scale=(0.08, 1.0), interpolation=InterpolationMode.BILINEAR
            ),
            RandomHorizontalFlipWithRes(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    dataset = ImageFolder_FKD_MIX(
        fkd_path=str(args.source_fkd), mode="fkd_load",
        args_epoch=args.epochs, args_bs=args.batch_size,
        root=str(args.image_root), transform=transform,
    )
    if len(dataset) != 10 * args.ipc or len(dataset.classes) != 10:
        raise RuntimeError(f"unexpected dataset: {len(dataset)} images, {len(dataset.classes)} classes")
    generator = torch.Generator().manual_seed(args.fkd_seed)
    sampler = torch.utils.data.RandomSampler(dataset, generator=generator)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=False,
    )
    mix_args = SimpleNamespace(mode="fkd_load", mix_type="cutmix", cutmix=1.0, mixup=0.8)
    args.output_fkd.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running", "dataset_name": "imagenet-nette", "ipc": args.ipc,
        "synthetic_data_path": str(args.image_root.resolve()),
        "fkd_path": str(args.output_fkd.resolve()),
        "source_fkd": str(args.source_fkd.resolve()),
        "source_fkd_manifest_sha256": sha256(args.source_fkd / "relabel_manifest.json"),
        "metadata_replay": ["batch order", "RRC coords", "flip", "CutMix index", "CutMix lambda", "CutMix bbox"],
        "teacher_path": str(args.teacher.resolve()), "teacher_sha256": sha256(args.teacher),
        "teacher_mode": "train", "epochs": args.epochs, "batch_size": args.batch_size,
        "teacher_forward_split": [5, 5], "workers": args.workers,
        "persistent_workers": False, "seed": args.seed, "fkd_seed": args.fkd_seed,
        "temperature": 20.0, "temperature_role": "post-eval softmax",
        "min_scale_crops": 0.08, "max_scale_crops": 1.0,
        "crop_interpolation": "bilinear", "mix_type": "cutmix", "cutmix_alpha": 1.0,
        "use_fp16": True, "rescored_from_actual_replayed_views": True,
    }
    atomic_json(manifest, manifest_path)
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        epoch_dir = args.output_fkd / f"epoch_{epoch}"
        epoch_dir.mkdir(exist_ok=True)
        started = time.time()
        batches = 0
        for batch_index, batch_data in enumerate(loader):
            images, _, flip_status, coords_status = batch_data[0]
            mix_index, mix_lam, mix_bbox = batch_data[1:4]
            images = images.cuda(non_blocking=True)
            mixed, _, _, _ = mix_aug(images, mix_args, mix_index, mix_lam, mix_bbox)
            first = teacher(mixed[:5])
            second = teacher(mixed[5:])
            logits = torch.cat((first, second), dim=0).half().cpu()
            torch.save(
                [coords_status, flip_status, mix_index.cpu(), mix_lam, mix_bbox, logits],
                epoch_dir / f"batch_{batch_index}.tar",
            )
            batches += 1
        if batches != args.ipc:
            raise RuntimeError(f"epoch {epoch}: {batches} batches != IPC {args.ipc}")
        print(f"epoch {epoch}/{args.epochs - 1} batches={batches} seconds={time.time()-started:.3f}", flush=True)
    files = list(args.output_fkd.glob("epoch_*/batch_*.tar"))
    if len(files) != expected_batches:
        raise RuntimeError(f"FKD files {len(files)} != {expected_batches}")
    manifest["status"] = "complete"
    manifest["batch_files"] = len(files)
    atomic_json(manifest, manifest_path)
    print(json.dumps({"status": "complete", "fkd": str(args.output_fkd.resolve()), "batch_files": len(files)}))


if __name__ == "__main__":
    main()
