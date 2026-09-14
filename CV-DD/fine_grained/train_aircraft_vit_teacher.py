"""Train pinned ViT-B/16 or TransFG-B/16 Aircraft teachers for exactly 10k updates."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vit_teacher_common import (
    AircraftRawTrain,
    FixedUpdateBatchSampler,
    StandardAircraftTest,
    atomic_json,
    build_teacher,
    class_mapping_digest,
    inference_logits,
    learning_rate,
    sha256_file,
    teacher_logits_and_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("vit", "transfg"))
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--train-index-root", required=True, type=Path)
    parser.add_argument("--raw-image-root", required=True, type=Path)
    parser.add_argument("--test-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--total-updates", default=10_000, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--checkpoint-every", default=500, type=int)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def save_checkpoint(model, optimizer, step: int, path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "manifest": manifest,
        },
        temporary,
    )
    os.replace(temporary, path)


@torch.no_grad()
def evaluate(model, kind: str, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    count = correct = 0
    nll_sum = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = inference_logits(model, kind, images)
        logits = logits.float()
        nll_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
        correct += int(logits.argmax(1).eq(labels).sum())
        count += labels.numel()
    return {"images": count, "top1": 100.0 * correct / count, "nll": nll_sum / count}


def main() -> None:
    args = parse_args()
    if args.batch_size != 64:
        raise RuntimeError("this protocol requires a true batch size of 64")
    if args.total_updates != 10_000 and not args.preflight_only:
        raise RuntimeError("formal protocol requires 10,000 optimizer updates")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    model, architecture = build_teacher(
        args.kind, args.source_root, weights_path=args.weights, checkpoint_path=None
    )
    train_set = AircraftRawTrain(args.train_index_root, args.raw_image_root, args.seed)
    test_set = StandardAircraftTest(args.test_root)
    if len(train_set) != 6667 or len(test_set) != 3333:
        raise RuntimeError(f"unexpected Aircraft split sizes: {len(train_set)}, {len(test_set)}")
    if train_set.classes != test_set.classes or len(train_set.classes) != 100:
        raise RuntimeError("train/test class mapping mismatch")
    manifest = {
        "status": "preflight" if args.preflight_only else "running",
        "protocol": "aircraft_vit_teacher_v1",
        "kind": args.kind,
        "seed": args.seed,
        "architecture": architecture,
        "pretrained_path": str(args.weights.resolve()),
        "pretrained_sha256": sha256_file(args.weights),
        "train_index_root": str(args.train_index_root.resolve()),
        "train_raw_image_root": str(args.raw_image_root.resolve()),
        "test_root": str(args.test_root.resolve()),
        "train_images": len(train_set),
        "test_images": len(test_set),
        "classes": len(train_set.classes),
        "class_mapping_sha256": class_mapping_digest(train_set.classes),
        "optimizer": {"name": "SGD", "peak_lr": 0.03, "momentum": 0.9, "weight_decay": 0.0, "nesterov": False},
        "schedule": {"unit": "optimizer_update", "warmup_updates": 500, "type": "cosine", "total_updates": args.total_updates, "final_lr": 0.0},
        "batch_size": 64,
        "gradient_accumulation": 1,
        "gradient_clip_global_norm": 1.0,
        "amp": "native_bfloat16",
        "loss_compute_dtype": "float32",
        "classification_loss": "cross_entropy_T1_no_label_smoothing",
        "contrast_loss": ({"coefficient": 1.0, "margin": 0.4, "pair_scope": 64, "normalization": "batch_size_squared"} if args.kind == "transfg" else None),
        "train_transform": ["RGB", "Resize(256,256,bilinear)", "RandomCrop(224,224)", "HorizontalFlip(p=0.5)", "float32_[0,1]", "ImageNet_mean_std"],
        "test_transform": ["existing_standard_224_warp", "RGB", "float32_[0,1]", "ImageNet_mean_std"],
        "augmentation_rng": "stable_sha256_per_image_update_slot; isolated from sampler/model RNG",
        "sample_order": "fixed torch randperm stream seed42; drop_last within each permutation",
        "head_initialization": "explicit_all_zero",
        "checkpoint_selection": "Final@10000",
    }
    atomic_json(manifest, args.output_dir / "training_manifest.json")
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.03, momentum=0.9, weight_decay=0.0, nesterov=False
    )
    start_update = 0
    latest = args.output_dir / "latest.pth"
    if latest.is_file() and not args.preflight_only:
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_update = int(payload["step"])
        torch.set_rng_state(payload["torch_rng_state"])
        torch.cuda.set_rng_state(payload["cuda_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        random.setstate(payload["python_rng_state"])
    total_updates = 1 if args.preflight_only else args.total_updates
    sampler = FixedUpdateBatchSampler(
        len(train_set), args.batch_size, total_updates, start_update, args.seed
    )
    train_loader = DataLoader(
        train_set,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model.train()
    started = time.time()
    for zero_based, (images, labels) in enumerate(train_loader, start=start_update):
        update_number = zero_based + 1
        lr = learning_rate(update_number, total=args.total_updates)
        for group in optimizer.param_groups:
            group["lr"] = lr
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, loss, parts = teacher_logits_and_loss(model, args.kind, images, labels)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update_number == 1 or update_number % 100 == 0:
            elapsed = time.time() - started
            record = {
                "update": update_number,
                "lr": lr,
                "loss": float(loss.detach()),
                **parts,
                "batch_top1": float(logits.detach().argmax(1).eq(labels).float().mean() * 100),
                "gradient_norm_before_clip": float(gradient_norm),
                "elapsed_seconds": elapsed,
                "updates_per_second": (update_number - start_update) / elapsed,
                "max_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            print(json.dumps(record), flush=True)
        if args.preflight_only:
            manifest["status"] = "preflight_complete"
            manifest["preflight"] = {
                "loss": float(loss.detach()),
                "gradient_norm_before_clip": float(gradient_norm),
                "max_cuda_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                "batch_size": labels.numel(),
            }
            atomic_json(manifest, args.output_dir / "training_manifest.json")
            return
        if update_number % args.checkpoint_every == 0 or update_number == args.total_updates:
            save_checkpoint(model, optimizer, update_number, latest, manifest)
    test_loader = DataLoader(
        test_set, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True
    )
    metrics = evaluate(model, args.kind, test_loader, device)
    final_path = args.output_dir / "final_step10000.pth"
    save_checkpoint(model, optimizer, args.total_updates, final_path, manifest)
    manifest["status"] = "complete"
    manifest["test"] = metrics
    manifest["final_checkpoint"] = str(final_path.resolve())
    manifest["final_checkpoint_sha256"] = sha256_file(final_path)
    manifest["wall_seconds"] = time.time() - started
    atomic_json(manifest, args.output_dir / "training_manifest.json")
    print(json.dumps({"status": "complete", "kind": args.kind, "test": metrics}), flush=True)


if __name__ == "__main__":
    main()
