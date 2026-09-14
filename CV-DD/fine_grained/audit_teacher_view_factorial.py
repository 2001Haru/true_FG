"""Four-cell full-frame/RRC x no-CutMix/CutMix teacher robustness audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from vit_teacher_common import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    atomic_json,
    build_teacher,
    inference_logits,
    sha256_file,
)


AIRCRAFT_MEAN = (0.4865, 0.5177, 0.5425)
AIRCRAFT_STD = (0.2124, 0.2051, 0.2375)
CELLS = ("fullframe", "rrc_only", "cutmix_only", "rrc_cutmix")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeler", required=True, choices=("resnet_bssl", "resnet_eval", "vit", "transfg"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--rrc-fkd", required=True, type=Path)
    parser.add_argument("--cutmix-fkd", required=True, type=Path)
    parser.add_argument("--sample-epochs", nargs="+", type=int, default=tuple(range(0, 400, 20)) + (399,))
    parser.add_argument("--batch-size", default=20, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_model(args):
    if args.labeler.startswith("resnet"):
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        if any(name.startswith("module.") for name in state):
            state = {name.removeprefix("module."): value for name, value in state.items()}
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 100)
        model.load_state_dict(state, strict=True)
        audit = {"architecture": "torchvision_resnet18", "normalization": "Aircraft"}
        if args.labeler == "resnet_bssl":
            model.train()
        else:
            model.eval()
        return model.cuda(), audit, AIRCRAFT_MEAN, AIRCRAFT_STD
    if args.source_root is None:
        raise ValueError("--source-root is required for ViT-family labelers")
    kind = args.labeler
    model, audit = build_teacher(
        kind, args.source_root, weights_path=None, checkpoint_path=args.checkpoint
    )
    return model.cuda().eval(), audit, IMAGENET_MEAN, IMAGENET_STD


def crop_from_coords(image: Image.Image, coords) -> Image.Image:
    values = torch.as_tensor(coords, dtype=torch.float32)
    top = float(values[0]) * image.size[1]
    left = float(values[1]) * image.size[0]
    height = float(values[2]) * image.size[1]
    width = float(values[3]) * image.size[0]
    return TF.resized_crop(
        image, top, left, height, width, (224, 224), InterpolationMode.BILINEAR
    )


def build_pixels(samples, indices, coords, flips):
    images = []
    for row, sample_index in enumerate(indices):
        path, _ = samples[sample_index]
        with Image.open(path) as source:
            image = crop_from_coords(source.convert("RGB"), coords[row])
        if bool(flips[row]):
            image = TF.hflip(image)
        images.append(TF.pil_to_tensor(image).float().div_(255.0))
    return torch.stack(images)


def forward(model, labeler, pixels, mean, std):
    images = TF.normalize(pixels.cuda(non_blocking=True), mean, std)
    if labeler.startswith("resnet"):
        return torch.cat((model(images[:10]), model(images[10:])), dim=0).float()
    return inference_logits(model, labeler, images).float()


def add_metrics(destination, logits, receiver, donor, donor_weight):
    logp1 = logits.log_softmax(1)
    p1 = logp1.exp()
    logp20 = (logits / 20.0).log_softmax(1)
    p20 = logp20.exp()
    row = torch.arange(logits.shape[0], device=logits.device)
    receiver = receiver.to(logits.device)
    donor = donor.to(logits.device)
    receiver_weight = 1.0 - donor_weight
    prediction = logits.argmax(1)
    destination["valid_source_argmax"].append(
        (prediction.eq(receiver) | prediction.eq(donor)).float().cpu()
    )
    destination["receiver_argmax"].append(prediction.eq(receiver).float().cpu())
    destination["source_target_ce_T1"].append(
        -(receiver_weight * logp1[row, receiver] + donor_weight * logp1[row, donor]).cpu()
    )
    destination["source_target_ce_T20"].append(
        -(receiver_weight * logp20[row, receiver] + donor_weight * logp20[row, donor]).cpu()
    )
    destination["source_target_probability_T1"].append(
        (receiver_weight * p1[row, receiver] + donor_weight * p1[row, donor]).cpu()
    )
    destination["entropy_T1"].append((-(p1 * logp1).sum(1)).cpu())
    destination["entropy_T20"].append((-(p20 * logp20).sum(1)).cpu())
    destination["logit_std"].append(logits.std(1, unbiased=False).cpu())


def summarize(chunks):
    values = torch.cat(chunks).double()
    return {
        "count": values.numel(),
        "mean": float(values.mean()),
        "sample_sd": float(values.std(unbiased=True)),
        "p10": float(values.quantile(0.10)),
        "median": float(values.quantile(0.50)),
        "p90": float(values.quantile(0.90)),
    }


@torch.no_grad()
def main():
    args = parse_args()
    torch.set_num_threads(1)
    dataset = datasets.ImageFolder(args.images)
    if len(dataset) != 300 or len(dataset.classes) != 100:
        raise RuntimeError(f"expected Aircraft IPC3 tree, got {len(dataset)} / {len(dataset.classes)}")
    requested = set(args.sample_epochs)
    generator = torch.Generator().manual_seed(args.seed)
    orders = {}
    for epoch in range(max(requested) + 1):
        order = torch.randperm(len(dataset), generator=generator).tolist()
        torch.randperm(len(dataset), generator=generator)[:0]
        if epoch in requested:
            orders[epoch] = order
    model, architecture, mean, std = load_model(args)
    metrics = {
        cell: {name: [] for name in (
            "valid_source_argmax", "receiver_argmax", "source_target_ce_T1",
            "source_target_ce_T20", "source_target_probability_T1",
            "entropy_T1", "entropy_T20", "logit_std",
        )}
        for cell in CELLS
    }
    metadata_checks = {"full_coords_nonfull": 0, "flip_mismatches": 0, "unexpected_rrc_mix": 0}
    for epoch in sorted(requested):
        order = orders[epoch]
        for batch_index, offset in enumerate(range(0, len(dataset), args.batch_size)):
            indices = order[offset : offset + args.batch_size]
            receiver = torch.tensor([dataset.samples[index][1] for index in indices], dtype=torch.long)
            rrc = torch.load(args.rrc_fkd / f"epoch_{epoch}" / f"batch_{batch_index}.tar", map_location="cpu", weights_only=False)
            cutmix = torch.load(args.cutmix_fkd / f"epoch_{epoch}" / f"batch_{batch_index}.tar", map_location="cpu", weights_only=False)
            if rrc[2] is not None:
                metadata_checks["unexpected_rrc_mix"] += 1
            full_expected = torch.tensor([0.0, 0.0, 1.0, 1.0])
            metadata_checks["full_coords_nonfull"] += sum(
                not torch.allclose(torch.as_tensor(coords).float(), full_expected, atol=1e-7, rtol=0)
                for coords in cutmix[0]
            )
            metadata_checks["flip_mismatches"] += sum(bool(a) != bool(b) for a, b in zip(rrc[1], cutmix[1]))
            # Freeze the RRC-arm flip stream for every cell so flip is not a fifth factor.
            flips = rrc[1]
            full_coords = [full_expected] * len(indices)
            full_pixels = build_pixels(dataset.samples, indices, full_coords, flips)
            rrc_pixels = build_pixels(dataset.samples, indices, rrc[0], flips)
            mix_index = cutmix[2].long()
            x1, y1, x2, y2 = (int(value) for value in cutmix[4])
            donor_weight = ((x2 - x1) * (y2 - y1)) / float(224 * 224)
            donor = receiver[mix_index]
            mixed_full = full_pixels.clone()
            mixed_full[:, :, x1:x2, y1:y2] = full_pixels[mix_index, :, x1:x2, y1:y2]
            mixed_rrc = rrc_pixels.clone()
            mixed_rrc[:, :, x1:x2, y1:y2] = rrc_pixels[mix_index, :, x1:x2, y1:y2]
            cell_data = {
                "fullframe": (full_pixels, receiver, 0.0),
                "rrc_only": (rrc_pixels, receiver, 0.0),
                "cutmix_only": (mixed_full, donor, donor_weight),
                "rrc_cutmix": (mixed_rrc, donor, donor_weight),
            }
            for cell, (pixels, donor_labels, weight) in cell_data.items():
                logits = forward(model, args.labeler, pixels, mean, std)
                add_metrics(metrics[cell], logits, receiver, donor_labels, weight)
    summarized = {
        cell: {name: summarize(chunks) for name, chunks in cell_metrics.items()}
        for cell, cell_metrics in metrics.items()
    }
    def mean(cell, metric):
        return summarized[cell][metric]["mean"]
    effects = {}
    for metric in ("valid_source_argmax", "source_target_ce_T1", "source_target_probability_T1", "entropy_T1"):
        effects[metric] = {
            "rrc_at_no_cutmix": mean("rrc_only", metric) - mean("fullframe", metric),
            "cutmix_at_no_rrc": mean("cutmix_only", metric) - mean("fullframe", metric),
            "rrc_at_cutmix": mean("rrc_cutmix", metric) - mean("cutmix_only", metric),
            "cutmix_at_rrc": mean("rrc_cutmix", metric) - mean("rrc_only", metric),
            "factorial_interaction": mean("rrc_cutmix", metric) - mean("rrc_only", metric) - mean("cutmix_only", metric) + mean("fullframe", metric),
        }
    result = {
        "status": "complete",
        "labeler": args.labeler,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "architecture": architecture,
        "images": str(args.images.resolve()),
        "sample_epochs": sorted(requested),
        "views_per_cell": len(requested) * len(dataset),
        "cell_construction": {
            "rrc_coords": str(args.rrc_fkd.resolve()),
            "cutmix_permutation_bbox": str(args.cutmix_fkd.resolve()),
            "flip_stream": "rrc_no_cutmix metadata, shared by all four cells",
            "rrc_scale": [0.08, 1.0],
        },
        "metadata_checks": metadata_checks,
        "cells": summarized,
        "effects": effects,
    }
    atomic_json(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
