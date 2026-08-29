import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torchvision

from config import get_dataset


def main() -> None:
    parser = argparse.ArgumentParser("Audit FKD teacher-logit semantics against ImageFolder classes")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--fkd-dir", required=True, type=Path)
    parser.add_argument("--epochs", nargs="+", type=int, default=(0, 100, 200, 399))
    parser.add_argument("--fkd-seed", type=int, default=42)
    parser.add_argument("--temperatures", nargs="+", type=float, default=(1.0, 3.0, 20.0))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = get_dataset(args.dataset_name)
    dataset = torchvision.datasets.ImageFolder(args.image_dir)
    if len(dataset.classes) != cfg.classes:
        raise RuntimeError(f"ImageFolder classes={len(dataset.classes)} != {cfg.classes}")
    targets = torch.tensor(dataset.targets, dtype=torch.long)
    selected_epochs = set(args.epochs)
    if min(selected_epochs) < 0:
        raise ValueError("epochs must be nonnegative")

    generator = torch.Generator().manual_seed(args.fkd_seed)
    sampler = torch.utils.data.RandomSampler(dataset, generator=generator)
    batch_sampler = torch.utils.data.BatchSampler(
        sampler, batch_size=cfg.fkd_batch_size, drop_last=False
    )
    results = []
    for epoch in range(max(selected_epochs) + 1):
        batches = [list(indices) for indices in batch_sampler]
        if epoch not in selected_epochs:
            continue
        logits_parts = []
        target_parts = []
        donor_target_parts = []
        original_fraction_parts = []
        mix_lams = []
        for batch_index, indices in enumerate(batches):
            path = args.fkd_dir / f"epoch_{epoch}" / f"batch_{batch_index}.tar"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            logits = payload[5]
            if logits.ndim == 3:
                logits = logits[-1]
            logits = logits.float()
            expected_targets = targets[indices]
            if logits.shape != (len(indices), cfg.classes):
                raise RuntimeError(
                    f"unexpected logits shape {tuple(logits.shape)} at {path}"
                )
            logits_parts.append(logits)
            target_parts.append(expected_targets)
            rand_index = torch.as_tensor(payload[2], dtype=torch.long)
            donor_target_parts.append(expected_targets[rand_index])
            bbx1, bby1, bbx2, bby2 = [int(value) for value in payload[4]]
            replaced_fraction = ((bbx2 - bbx1) * (bby2 - bby1)) / (224 * 224)
            original_fraction_parts.append(torch.full(
                (len(indices),), 1.0 - replaced_fraction, dtype=torch.float32
            ))
            mix_lams.append(float(payload[3]))
        logits = torch.cat(logits_parts)
        labels = torch.cat(target_parts)
        donor_labels = torch.cat(donor_target_parts)
        original_fractions = torch.cat(original_fraction_parts)
        top5 = logits.topk(5, dim=1).indices
        top1_accuracy = 100.0 * top5[:, 0].eq(labels).float().mean().item()
        top5_accuracy = 100.0 * top5.eq(labels[:, None]).any(1).float().mean().item()
        dominant_labels = torch.where(
            original_fractions >= 0.5, labels, donor_labels
        )
        dominant_accuracy = 100.0 * top5[:, 0].eq(dominant_labels).float().mean().item()
        either_accuracy = 100.0 * (
            top5[:, 0].eq(labels) | top5[:, 0].eq(donor_labels)
        ).float().mean().item()
        target_logits = logits.gather(1, labels[:, None]).squeeze(1)
        mask = torch.nn.functional.one_hot(labels, cfg.classes).bool()
        strongest_other = logits.masked_fill(mask, float("-inf")).max(1).values

        temperature_stats = {}
        for temperature in args.temperatures:
            probabilities = torch.softmax(logits / temperature, dim=1)
            target_probability = probabilities.gather(1, labels[:, None]).squeeze(1)
            donor_probability = probabilities.gather(1, donor_labels[:, None]).squeeze(1)
            same_component = labels.eq(donor_labels)
            either_probability_mass = torch.where(
                same_component,
                target_probability,
                target_probability + donor_probability,
            )
            weighted_component_probability = (
                original_fractions * target_probability
                + (1.0 - original_fractions) * donor_probability
            )
            mixed_target_cross_entropy = -(
                original_fractions * target_probability.clamp_min(1e-12).log()
                + (1.0 - original_fractions) * donor_probability.clamp_min(1e-12).log()
            )
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)
            key = str(int(temperature)) if temperature.is_integer() else str(temperature)
            temperature_stats[key] = {
                "mean_target_probability": target_probability.mean().item(),
                "median_target_probability": target_probability.median().item(),
                "mean_either_component_probability_mass": either_probability_mass.mean().item(),
                "mean_area_weighted_component_probability": weighted_component_probability.mean().item(),
                "mean_mixed_target_cross_entropy": mixed_target_cross_entropy.mean().item(),
                "mean_entropy_nats": entropy.mean().item(),
                "mean_normalized_entropy": (
                    entropy.mean().item() / math.log(cfg.classes)
                ),
            }
        results.append({
            "epoch": epoch,
            "images": len(labels),
            "teacher_top1_vs_hard_class": top1_accuracy,
            "teacher_top5_vs_hard_class": top5_accuracy,
            "teacher_top1_vs_dominant_cutmix_class": dominant_accuracy,
            "teacher_top1_in_either_cutmix_component": either_accuracy,
            "mean_target_logit_margin": (target_logits - strongest_other).mean().item(),
            "median_target_logit_margin": (target_logits - strongest_other).median().item(),
            "mean_cutmix_lambda": float(np.mean(mix_lams)),
            "mean_actual_original_area_fraction": original_fractions.mean().item(),
            "temperature": temperature_stats,
        })

    payload = {
        "status": "complete",
        "dataset": cfg.name,
        "image_dir": str(args.image_dir.resolve()),
        "fkd_dir": str(args.fkd_dir.resolve()),
        "fkd_seed": args.fkd_seed,
        "cutmix_teacher_input": "The teacher input alias is modified in-place by CutMix; teacher and student replay the same mixed tensor.",
        "epochs": results,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
