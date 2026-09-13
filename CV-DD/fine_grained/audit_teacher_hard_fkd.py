"""Audit per-view Teacher-argmax targets in an unmixed FKD trajectory."""
import argparse
import json
import math
import os
from pathlib import Path

import torch


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff", ".webp"}
AREA_EDGES = (0.0, 0.10, 0.25, 0.50, 0.75, 1.000001)


def summarize(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "count": len(values),
        "mean": float(tensor.mean()) if len(values) else None,
        "q10": float(torch.quantile(tensor, 0.10)) if len(values) else None,
        "median": float(torch.quantile(tensor, 0.50)) if len(values) else None,
        "q90": float(torch.quantile(tensor, 0.90)) if len(values) else None,
    }


def main():
    # Thousands of tiny 20x100 softmax calls are faster and less disruptive
    # without spawning a full CPU thread team for every saved FKD batch.
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--fkd", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classes", type=int, default=100)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    class_dirs = sorted(path for path in args.images.iterdir() if path.is_dir())
    if len(class_dirs) != args.classes:
        raise RuntimeError(f"expected {args.classes} class dirs, found {len(class_dirs)}")
    targets = []
    for class_id, directory in enumerate(class_dirs):
        files = sorted(path for path in directory.rglob("*")
                       if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        targets.extend([class_id] * len(files))
    number_images = len(targets)
    if number_images != 300:
        raise RuntimeError(f"expected Aircraft IPC3 tree with 300 images, found {number_images}")

    generator = torch.Generator().manual_seed(args.seed)
    original_count = [0] * args.classes
    mismatch_count = [0] * args.classes
    pseudo_histogram = [0] * args.classes
    area_total = [0] * (len(AREA_EDGES) - 1)
    area_mismatch = [0] * (len(AREA_EDGES) - 1)
    all_confidence, changed_confidence, unchanged_confidence = [], [], []
    total = mismatches = 0

    for epoch in range(args.epochs):
        order = torch.randperm(number_images, generator=generator).tolist()
        # torch.utils.data.RandomSampler consumes one additional randperm even
        # when num_samples % len(dataset) == 0; its final [:0] is empty but
        # advances the generator and therefore changes every later epoch.
        torch.randperm(number_images, generator=generator)[:0]
        for batch_index, offset in enumerate(range(0, number_images, args.batch_size)):
            path = args.fkd / f"epoch_{epoch}" / f"batch_{batch_index}.tar"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if len(payload) not in (6, 7) or payload[2] is not None:
                raise RuntimeError(f"expected unmixed FKD payload at {path}")
            coords, logits = payload[0], payload[5].float()
            batch_order = order[offset:offset + logits.shape[0]]
            true = torch.tensor([targets[index] for index in batch_order], dtype=torch.long)
            probabilities = logits.softmax(dim=1)
            confidence, pseudo = probabilities.max(dim=1)
            changed = pseudo.ne(true)
            areas = torch.as_tensor(coords, dtype=torch.float32)[:, 2:4].prod(dim=1)
            for row in range(logits.shape[0]):
                y = int(true[row]); p = int(pseudo[row]); is_changed = bool(changed[row])
                area = float(areas[row]); conf = float(confidence[row])
                bin_id = min(len(AREA_EDGES) - 2,
                             max(0, next((index for index in range(len(AREA_EDGES) - 1)
                                          if AREA_EDGES[index] <= area < AREA_EDGES[index + 1]),
                                         len(AREA_EDGES) - 2)))
                total += 1; original_count[y] += 1; pseudo_histogram[p] += 1
                area_total[bin_id] += 1; all_confidence.append(conf)
                if is_changed:
                    mismatches += 1; mismatch_count[y] += 1; area_mismatch[bin_id] += 1
                    changed_confidence.append(conf)
                else:
                    unchanged_confidence.append(conf)

    per_class = [{
        "class_id": class_id,
        "class_name": class_dirs[class_id].name,
        "views": original_count[class_id],
        "mismatches": mismatch_count[class_id],
        "mismatch_rate": mismatch_count[class_id] / original_count[class_id],
    } for class_id in range(args.classes)]
    area = [{
        "lower": AREA_EDGES[index], "upper": min(1.0, AREA_EDGES[index + 1]),
        "views": area_total[index], "mismatches": area_mismatch[index],
        "mismatch_rate": area_mismatch[index] / area_total[index] if area_total[index] else None,
    } for index in range(len(area_total))]
    result = {
        "status": "complete", "images": str(args.images.resolve()),
        "fkd": str(args.fkd.resolve()), "epochs": args.epochs,
        "views": total, "mismatches": mismatches,
        "mismatch_rate": mismatches / total,
        "per_original_class": per_class,
        "crop_area_bins": area,
        "teacher_t1_max_probability": {
            "all": summarize(all_confidence),
            "replaced": summarize(changed_confidence),
            "unchanged": summarize(unchanged_confidence),
        },
        "teacher_hard_target_histogram": pseudo_histogram,
        "target_histogram_min": min(pseudo_histogram),
        "target_histogram_max": max(pseudo_histogram),
        "target_histogram_cv": (float(torch.tensor(pseudo_histogram, dtype=torch.float64).std(unbiased=True) /
                                      torch.tensor(pseudo_histogram, dtype=torch.float64).mean())),
        "sampler": "exact torch RandomSampler replay, including empty remainder randperm; seed 42",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({key: result[key] for key in
                      ("status", "views", "mismatches", "mismatch_rate",
                       "target_histogram_min", "target_histogram_max", "target_histogram_cv")}, indent=2))


if __name__ == "__main__":
    main()
