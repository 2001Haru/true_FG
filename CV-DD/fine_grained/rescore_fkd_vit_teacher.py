"""Replay the R0 geometry/CutMix schedule and rescore it with a ViT-family teacher."""

from __future__ import annotations

import argparse
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
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data._utils.fetch import _MapDatasetFetcher
from torchvision.transforms import InterpolationMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relabel.utils_fkd import (  # noqa: E402
    ComposeWithCoords,
    ImageFolder_FKD_MIX,
    RandomHorizontalFlipWithRes,
    RandomResizedCropWithCoords,
    mix_aug,
)

from vit_teacher_common import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    atomic_json,
    build_teacher,
    inference_logits,
    sha256_file,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("vit", "transfg"))
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--source-fkd", required=True, type=Path)
    parser.add_argument("--output-fkd", required=True, type=Path)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--batch-size", default=20, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--fkd-seed", default=42, type=int)
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.epochs != 400 or args.batch_size != 20:
        raise RuntimeError("R0 v2 replay is fixed at 400 epochs and batch20")
    manifest_path = args.output_fkd / "relabel_manifest.json"
    expected_batches = args.epochs * math.ceil(300 / args.batch_size)
    if args.skip_completed and manifest_path.is_file():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current.get("status") == "complete" and len(list(args.output_fkd.glob("epoch_*/batch_*.tar"))) == expected_batches:
            print(json.dumps({"status": "already_complete", "fkd": str(args.output_fkd)}))
            return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, architecture = build_teacher(
        args.kind, args.source_root, weights_path=None, checkpoint_path=args.checkpoint
    )
    model.cuda().eval()
    transform = ComposeWithCoords(
        transforms=[
            RandomResizedCropWithCoords(
                size=224, scale=(0.08, 1.0), interpolation=InterpolationMode.BILINEAR
            ),
            RandomHorizontalFlipWithRes(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    dataset = ImageFolder_FKD_MIX(
        fkd_path=str(args.source_fkd),
        mode="fkd_load",
        args_epoch=args.epochs,
        args_bs=args.batch_size,
        root=str(args.image_root),
        transform=transform,
    )
    if len(dataset) != 300 or len(dataset.classes) != 100:
        raise RuntimeError(f"unexpected R0 dataset: {len(dataset)} images, {len(dataset.classes)} classes")
    sampler = torch.utils.data.RandomSampler(
        dataset, generator=torch.Generator().manual_seed(args.fkd_seed)
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=False,
    )
    mix_args = SimpleNamespace(mode="fkd_load", mix_type="cutmix", cutmix=1.0, mixup=0.8)
    args.output_fkd.mkdir(parents=True, exist_ok=True)
    source_manifest = args.source_fkd / "relabel_manifest.json"
    manifest = {
        "status": "running",
        "protocol": "aircraft_R0_v2_vit_labeler_v1",
        "kind": args.kind,
        "architecture": architecture,
        "image_root": str(args.image_root.resolve()),
        "source_fkd": str(args.source_fkd.resolve()),
        "source_fkd_manifest_sha256": sha256_file(source_manifest),
        "output_fkd": str(args.output_fkd.resolve()),
        "teacher_checkpoint": str(args.checkpoint.resolve()),
        "teacher_checkpoint_sha256": sha256_file(args.checkpoint),
        "teacher_mode": "eval",
        "teacher_forward_split": [10, 10],
        "normalization": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "metadata_replay": ["batch order", "RRC coords", "flip", "CutMix index", "CutMix lambda", "CutMix bbox"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "persistent_workers": False,
        "seed": args.seed,
        "fkd_seed": args.fkd_seed,
        "saved_values": "raw_logits_float16",
        "student_temperature": 20.0,
        "student_T_squared": False,
    }
    atomic_json(manifest, manifest_path)
    entropy_t1: list[torch.Tensor] = []
    entropy_t20: list[torch.Tensor] = []
    logit_std: list[torch.Tensor] = []
    split_full_max_abs = None
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
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                first = inference_logits(model, args.kind, mixed[:10])
                second = inference_logits(model, args.kind, mixed[10:])
                logits = torch.cat((first, second), dim=0)
                if split_full_max_abs is None:
                    full = inference_logits(model, args.kind, mixed)
                    split_full_max_abs = float((full.float() - logits.float()).abs().max())
            values = logits.float()
            for temperature, destination in ((1.0, entropy_t1), (20.0, entropy_t20)):
                log_probability = F.log_softmax(values / temperature, dim=1)
                probability = log_probability.exp()
                destination.append((-(probability * log_probability).sum(1)).cpu())
            logit_std.append(values.std(dim=1, unbiased=False).cpu())
            torch.save(
                [coords_status, flip_status, mix_index.cpu(), mix_lam, mix_bbox, values.half().cpu()],
                epoch_dir / f"batch_{batch_index}.tar",
            )
            batches += 1
        if batches != 15:
            raise RuntimeError(f"epoch {epoch}: {batches} batches != 15")
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(json.dumps({"epoch": epoch, "seconds": time.time() - started, "batches": batches}), flush=True)
    if len(list(args.output_fkd.glob("epoch_*/batch_*.tar"))) != expected_batches:
        raise RuntimeError("incomplete output FKD")
    stats = {}
    for name, chunks in (("entropy_T1_nats", entropy_t1), ("entropy_T20_nats", entropy_t20), ("per_view_logit_std", logit_std)):
        vector = torch.cat(chunks).float()
        stats[name] = {
            "count": vector.numel(),
            "mean": float(vector.mean()),
            "std": float(vector.std(unbiased=False)),
            "p01": float(vector.quantile(0.01)),
            "p50": float(vector.quantile(0.50)),
            "p99": float(vector.quantile(0.99)),
        }
    stats["split_vs_full_max_abs_logit_difference_first_batch"] = split_full_max_abs
    atomic_json(stats, args.output_fkd / "teacher_logit_audit.json")
    manifest["status"] = "complete"
    manifest["batch_files"] = expected_batches
    manifest["teacher_logit_audit"] = stats
    atomic_json(manifest, manifest_path)
    print(json.dumps({"status": "complete", "kind": args.kind, "stats": stats}), flush=True)


if __name__ == "__main__":
    main()
