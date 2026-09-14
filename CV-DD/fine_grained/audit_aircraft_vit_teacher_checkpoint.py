"""Evaluate a final Aircraft ViT-family teacher on deterministic train/test views."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import functional as TF

from vit_teacher_common import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    atomic_json,
    build_teacher,
    class_mapping_digest,
    inference_logits,
    sha256_file,
)


class DeterministicAircraft(Dataset):
    def __init__(self, index_root: Path, raw_root: Path | None):
        self.index = datasets.ImageFolder(index_root)
        self.classes = self.index.classes
        self.raw_root = raw_root

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        indexed_path, target = self.index.samples[index]
        path = self.raw_root / Path(indexed_path).name if self.raw_root else Path(indexed_path)
        with Image.open(path) as source:
            image = source.convert("RGB")
            if self.raw_root:
                image = image.resize((256, 256), Image.Resampling.BILINEAR)
                image = image.crop((16, 16, 240, 240))
            elif image.size != (224, 224):
                raise RuntimeError(f"standard test image is not 224x224: {path} {image.size}")
        tensor = TF.pil_to_tensor(image).float().div_(255.0)
        return TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD), target


@torch.no_grad()
def evaluate(model, kind, dataset, workers):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=workers, pin_memory=True)
    count = correct = 0
    nll_sum = entropy_sum = confidence_sum = 0.0
    bins = 15
    bin_count = torch.zeros(bins, dtype=torch.float64)
    bin_conf = torch.zeros(bins, dtype=torch.float64)
    bin_acc = torch.zeros(bins, dtype=torch.float64)
    class_total = torch.zeros(100, dtype=torch.long)
    class_correct = torch.zeros(100, dtype=torch.long)
    model.eval()
    for images, labels in loader:
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = inference_logits(model, kind, images)
        logits = logits.float()
        log_probability = F.log_softmax(logits, dim=1)
        probability = log_probability.exp()
        confidence, prediction = probability.max(1)
        is_correct = prediction.eq(labels)
        nll_sum += float(F.nll_loss(log_probability, labels, reduction="sum"))
        entropy_sum += float((-(probability * log_probability).sum(1)).sum())
        confidence_sum += float(confidence.sum())
        count += labels.numel()
        correct += int(is_correct.sum())
        class_total += torch.bincount(labels.cpu(), minlength=100)
        class_correct += torch.bincount(labels[is_correct].cpu(), minlength=100)
        bin_id = (confidence.mul(bins).long()).clamp_max(bins - 1).cpu()
        for current in range(bins):
            selected = bin_id.eq(current)
            if selected.any():
                bin_count[current] += int(selected.sum())
                bin_conf[current] += float(confidence.cpu()[selected].sum())
                bin_acc[current] += float(is_correct.cpu()[selected].sum())
    nonempty = bin_count.gt(0)
    ece = (((bin_conf[nonempty] / bin_count[nonempty]) - (bin_acc[nonempty] / bin_count[nonempty])).abs() * bin_count[nonempty]).sum() / count
    per_class = class_correct.double().div(class_total).mul(100.0)
    return {
        "images": count,
        "top1": 100.0 * correct / count,
        "nll": nll_sum / count,
        "mean_predictive_entropy_T1_nats": entropy_sum / count,
        "mean_max_probability_T1": confidence_sum / count,
        "ece_15bin": float(ece),
        "per_class_top1_min": float(per_class.min()),
        "per_class_top1_median": float(per_class.median()),
        "per_class_top1_max": float(per_class.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("vit", "transfg"))
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-index-root", required=True, type=Path)
    parser.add_argument("--raw-image-root", required=True, type=Path)
    parser.add_argument("--test-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", default=8, type=int)
    args = parser.parse_args()
    model, architecture = build_teacher(
        args.kind, args.source_root, weights_path=None, checkpoint_path=args.checkpoint
    )
    model.cuda().eval()
    train = DeterministicAircraft(args.train_index_root, args.raw_image_root)
    test = DeterministicAircraft(args.test_root, None)
    if train.classes != test.classes or class_mapping_digest(train.classes) != class_mapping_digest(test.classes):
        raise RuntimeError("train/test class mapping mismatch")
    result = {
        "status": "complete",
        "kind": args.kind,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "architecture": architecture,
        "train_view": "raw RGB -> warp Resize(256,256,bilinear) -> CenterCrop224 -> ImageNet normalize",
        "test_view": "existing standard 224 warp -> ImageNet normalize",
        "train": evaluate(model, args.kind, train, args.workers),
        "test": evaluate(model, args.kind, test, args.workers),
    }
    atomic_json(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
