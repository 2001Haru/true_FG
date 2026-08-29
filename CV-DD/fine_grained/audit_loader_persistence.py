from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, RandomSampler
from torchvision.transforms import InterpolationMode

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "validate"))

# Importing train_fkd installs the released-CV-DD batched-fetch compatibility
# patch used by the real student process.
import validate.train_fkd  # noqa: F401,E402
from fine_grained.config import get_dataset  # noqa: E402
from relabel.utils_fkd import (  # noqa: E402
    ComposeWithCoords,
    ImageFolder_FKD_MIX,
    RandomHorizontalFlipWithRes,
    RandomResizedCropWithCoords,
)


def update_digest(digest: hashlib._Hash, value) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode())
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        digest.update(b"ndarray")
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode())
        digest.update(len(value).to_bytes(8, "little"))
        for item in value:
            update_digest(digest, item)
    elif isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=str):
            update_digest(digest, key)
            update_digest(digest, value[key])
    else:
        digest.update(repr(value).encode())


def build_loader(args, persistent: bool):
    cfg = get_dataset(args.dataset_name)
    dataset = ImageFolder_FKD_MIX(
        fkd_path=str(args.fkd_dir), mode="fkd_load",
        args_epoch=args.epochs, args_bs=cfg.fkd_batch_size,
        root=str(args.image_dir),
        transform=ComposeWithCoords(transforms=[
            RandomResizedCropWithCoords(
                size=224, scale=(0.08, 1.0),
                interpolation=InterpolationMode.BILINEAR,
            ),
            RandomHorizontalFlipWithRes(),
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ]),
    )
    generator = torch.Generator().manual_seed(42)
    sampler = RandomSampler(dataset, generator=generator)
    loader = DataLoader(
        dataset, batch_size=cfg.fkd_batch_size, sampler=sampler,
        num_workers=args.workers, pin_memory=args.pin_memory,
        persistent_workers=persistent,
    )
    return dataset, loader


def run(args, persistent: bool) -> tuple[list[str], list[float]]:
    dataset, loader = build_loader(args, persistent)
    epoch_hashes = []
    seconds = []
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        started = time.perf_counter()
        digest = hashlib.sha256()
        for batch in loader:
            update_digest(digest, batch)
        seconds.append(time.perf_counter() - started)
        epoch_hashes.append(digest.hexdigest())
    return epoch_hashes, seconds


def main() -> None:
    parser = argparse.ArgumentParser("Prove FKD DataLoader persistent-worker equivalence")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--fkd-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive for a persistence audit")

    nonpersistent_hashes, nonpersistent_seconds = run(args, False)
    persistent_hashes, persistent_seconds = run(args, True)
    mismatches = [
        epoch for epoch, (left, right) in enumerate(
            zip(nonpersistent_hashes, persistent_hashes)
        ) if left != right
    ]
    payload = {
        "status": "equivalent" if not mismatches else "mismatch",
        "dataset": get_dataset(args.dataset_name).name,
        "image_dir": str(args.image_dir.resolve()),
        "fkd_dir": str(args.fkd_dir.resolve()),
        "epochs": args.epochs,
        "workers": args.workers,
        "pin_memory": args.pin_memory,
        "mismatch_epochs": mismatches,
        "epoch_sha256": nonpersistent_hashes,
        "nonpersistent_seconds": nonpersistent_seconds,
        "persistent_seconds": persistent_seconds,
        "nonpersistent_mean_seconds": float(np.mean(nonpersistent_seconds)),
        "persistent_mean_seconds_excluding_startup": float(
            np.mean(persistent_seconds[1:] or persistent_seconds)
        ),
        "speedup_excluding_persistent_startup": float(
            np.mean(nonpersistent_seconds) /
            np.mean(persistent_seconds[1:] or persistent_seconds)
        ),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if mismatches:
        raise RuntimeError(f"Persistent/nonpersistent mismatch at epochs {mismatches}")


if __name__ == "__main__":
    main()
