"""Compute entropy-biased ImageNette IPC10 selections and preflight audits."""

import argparse
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

from imagenette_entropy_protocol import (
    CALIBRATION_VIEW_SEED,
    CLASSES,
    DeterministicReleasedView,
    HOLDOUT_VIEW_SEED,
    IMAGE_SIZE,
    IPC,
    LAMBDAS,
    SELECTION_SEEDS,
    SELECTION_VIEWS,
    atomic_json,
    average_tie_percentiles,
    build_teacher,
    file_sha256,
    gumbel_noise,
    identity_seed,
    normalize,
    test_transform,
    validate_official_split,
)


class EntropyViewDataset(Dataset):
    def __init__(self, samples, data_root, random_namespace, phase_seed):
        self.samples = samples
        self.data_root = data_root
        self.random_namespace = random_namespace
        self.phase_seed = phase_seed
        self.view = DeterministicReleasedView()

    def __len__(self):
        return len(self.samples) * SELECTION_VIEWS

    def __getitem__(self, task_index):
        image_index, view_index = divmod(task_index, SELECTION_VIEWS)
        path, target = self.samples[image_index]
        relative = Path(path).relative_to(self.data_root).as_posix()
        seed = identity_seed(
            self.random_namespace, self.phase_seed, relative, view_index
        )
        with Image.open(path) as image:
            tensor = self.view(image, seed)
        return tensor, target, image_index


@torch.inference_mode()
def teacher_view_statistics(teacher, samples, data_root, random_namespace, phase_seed, args):
    dataset = EntropyViewDataset(samples, data_root, random_namespace, phase_seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=True,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    fields = {
        "entropy": torch.zeros(len(samples), dtype=torch.float64),
        "true_probability": torch.zeros(len(samples), dtype=torch.float64),
        "maximum_probability": torch.zeros(len(samples), dtype=torch.float64),
        "argmax_correct": torch.zeros(len(samples), dtype=torch.float64),
    }
    counts = torch.zeros(len(samples), dtype=torch.int64)
    for batch_index, (views, targets, indices) in enumerate(loader):
        views = normalize(views.cuda(non_blocking=True))
        targets = targets.cuda(non_blocking=True)
        logits = teacher(views).float()
        log_probabilities = F.log_softmax(logits, dim=1)
        probabilities = log_probabilities.exp()
        entropy = -(probabilities * log_probabilities).sum(dim=1) / math.log(CLASSES)
        confidence, prediction = probabilities.max(dim=1)
        true_probability = probabilities.gather(1, targets[:, None]).squeeze(1)
        values = {
            "entropy": entropy,
            "true_probability": true_probability,
            "maximum_probability": confidence,
            "argmax_correct": prediction.eq(targets).float(),
        }
        indices = indices.long()
        for key, value in values.items():
            fields[key].index_add_(0, indices, value.double().cpu())
        counts.index_add_(0, indices, torch.ones_like(indices))
        if batch_index % 100 == 0:
            print(
                f"phase_seed={phase_seed} views={min((batch_index + 1) * args.batch_size, len(dataset))}/{len(dataset)}",
                flush=True,
            )
    if not torch.equal(counts, torch.full_like(counts, SELECTION_VIEWS)):
        raise RuntimeError("not every image received exactly 16 entropy views")
    return {key: (value / counts).numpy() for key, value in fields.items()}


@torch.inference_mode()
def evaluate_teacher(teacher, test_dataset, workers):
    transformed = type(test_dataset)(test_dataset.root, transform=test_transform)
    loader = DataLoader(
        transformed,
        batch_size=256,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=True,
    )
    correct = total = 0
    for images, targets in loader:
        prediction = teacher(images.cuda(non_blocking=True)).argmax(dim=1).cpu()
        correct += prediction.eq(targets).sum().item()
        total += len(targets)
    return {"correct": correct, "images": total, "top1": 100.0 * correct / total}


def rankdata(values):
    return np.asarray(average_tie_percentiles(list(map(float, values))), dtype=np.float64)


def correlation(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def load_dino(cache_path, train_dataset, data_root):
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    split = cache["splits"]["train"]
    paths = split["relative_paths"]
    expected = [Path(path).relative_to(data_root).as_posix() for path, _ in train_dataset.samples]
    if paths != expected or split["targets"].tolist() != train_dataset.targets:
        raise RuntimeError("DINO cache train paths/targets differ from official ImageNette split")
    features = F.normalize(split["features"].float(), dim=1).numpy().astype(np.float64)
    return features, cache["metadata"]


def dino_geometry(features, targets):
    own_similarity = np.zeros(len(features))
    radial_distance = np.zeros(len(features))
    radial_percentile = np.zeros(len(features))
    for class_id in range(CLASSES):
        indices = np.flatnonzero(np.asarray(targets) == class_id)
        center = features[indices].sum(axis=0)
        center /= np.linalg.norm(center)
        own_similarity[indices] = features[indices] @ center
        radial_distance[indices] = 1.0 - own_similarity[indices]
        radial_percentile[indices] = rankdata(radial_distance[indices])
    return own_similarity, radial_distance, radial_percentile


def selection_metrics(indices, features, targets, rows):
    selected_set = set(indices)
    per_class_coverage = []
    per_class = []
    for class_id in range(CLASSES):
        class_indices = np.flatnonzero(np.asarray(targets) == class_id)
        selected = [index for index in class_indices if index in selected_set]
        nearest_distance = 1.0 - np.max(features[class_indices] @ features[selected].T, axis=1)
        per_class_coverage.append(float(nearest_distance.mean()))
        class_rows = [rows[index] for index in selected]
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_rows[0]["class_name"],
                "images": len(class_rows),
                "calibration_entropy_mean": statistics.mean(row["calibration_entropy"] for row in class_rows),
                "holdout_entropy_mean": statistics.mean(row["holdout_entropy"] for row in class_rows),
                "calibration_true_probability_mean": statistics.mean(row["calibration_true_probability"] for row in class_rows),
                "holdout_true_probability_mean": statistics.mean(row["holdout_true_probability"] for row in class_rows),
                "calibration_argmax_accuracy": statistics.mean(row["calibration_argmax_correct"] for row in class_rows),
                "holdout_argmax_accuracy": statistics.mean(row["holdout_argmax_correct"] for row in class_rows),
                "dino_radial_percentile_mean": statistics.mean(row["dino_radial_percentile"] for row in class_rows),
                "dino_full_class_mean_nearest_cosine_distance": float(nearest_distance.mean()),
            }
        )
    selected_rows = [rows[index] for index in indices]
    return {
        "selection_images": len(indices),
        "calibration_entropy_mean": statistics.mean(row["calibration_entropy"] for row in selected_rows),
        "calibration_entropy_min": min(row["calibration_entropy"] for row in selected_rows),
        "calibration_entropy_max": max(row["calibration_entropy"] for row in selected_rows),
        "holdout_entropy_mean": statistics.mean(row["holdout_entropy"] for row in selected_rows),
        "calibration_true_probability_mean": statistics.mean(row["calibration_true_probability"] for row in selected_rows),
        "holdout_true_probability_mean": statistics.mean(row["holdout_true_probability"] for row in selected_rows),
        "calibration_argmax_accuracy": statistics.mean(row["calibration_argmax_correct"] for row in selected_rows),
        "holdout_argmax_accuracy": statistics.mean(row["holdout_argmax_correct"] for row in selected_rows),
        "dino_radial_percentile_mean": statistics.mean(row["dino_radial_percentile"] for row in selected_rows),
        "dino_full_class_mean_nearest_cosine_distance": statistics.mean(per_class_coverage),
        "per_class": per_class,
    }


def contact_sheet(path, selected_rows, title):
    tile = 112
    caption = 20
    header = 28
    canvas = Image.new("RGB", (IPC * tile, header + CLASSES * (tile + caption)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 6), title, fill="black")
    grouped = defaultdict(list)
    for row in selected_rows:
        grouped[row["class_id"]].append(row)
    for class_id in range(CLASSES):
        group = sorted(grouped[class_id], key=lambda row: row["selection_score"], reverse=True)
        for column, row in enumerate(group):
            with Image.open(row["source_path"]) as image:
                image = image.convert("RGB")
                image.thumbnail((tile, tile))
                x = column * tile + (tile - image.width) // 2
                y0 = header + class_id * (tile + caption)
                y = y0 + (tile - image.height) // 2
                canvas.paste(image, (x, y))
            draw.text(
                (column * tile + 2, y0 + tile + 2),
                f"c{class_id} h={row['calibration_entropy']:.2f}",
                fill="black",
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--dino-cache", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--workers", default=16, type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    summary_path = output_root / "preflight" / "selection_preflight.json"
    if summary_path.exists() and not args.force:
        raise RuntimeError(f"preflight already exists; refusing overwrite: {summary_path}")
    train_dataset, test_dataset = validate_official_split(data_root)
    split_file_list_path = output_root / "preflight" / "official_split_files.json"
    split_file_list = {
        "status": "complete",
        "data_root": str(data_root),
        "class_to_idx": train_dataset.class_to_idx,
        "train": [
            {
                "relative_path": Path(path).relative_to(data_root).as_posix(),
                "class_id": int(target),
                "size_bytes": Path(path).stat().st_size,
            }
            for path, target in train_dataset.samples
        ],
        "test": [
            {
                "relative_path": Path(path).relative_to(data_root).as_posix(),
                "class_id": int(target),
                "size_bytes": Path(path).stat().st_size,
            }
            for path, target in test_dataset.samples
        ],
    }
    atomic_json(split_file_list_path, split_file_list)
    split_file_list_sha256 = file_sha256(split_file_list_path)
    teacher = build_teacher(args.teacher_checkpoint.resolve()).cuda().eval()
    teacher_test = evaluate_teacher(teacher, test_dataset, args.workers)
    calibration = teacher_view_statistics(
        teacher,
        train_dataset.samples,
        data_root,
        "imagenette-selection-calibration-view-v1",
        CALIBRATION_VIEW_SEED,
        args,
    )
    holdout = teacher_view_statistics(
        teacher,
        train_dataset.samples,
        data_root,
        "imagenette-selection-holdout-view-v1",
        HOLDOUT_VIEW_SEED,
        args,
    )
    features, dino_metadata = load_dino(args.dino_cache.resolve(), train_dataset, data_root)
    own_similarity, radial_distance, radial_percentile = dino_geometry(
        features, train_dataset.targets
    )

    entropy_percentile = np.zeros(len(train_dataset))
    for class_id in range(CLASSES):
        indices = np.flatnonzero(np.asarray(train_dataset.targets) == class_id)
        entropy_percentile[indices] = rankdata(calibration["entropy"][indices])
    rows = []
    for index, ((source_path, target), relative) in enumerate(
        zip(
            train_dataset.samples,
            [Path(path).relative_to(data_root).as_posix() for path, _ in train_dataset.samples],
        )
    ):
        rows.append(
            {
                "image_index": index,
                "class_id": int(target),
                "class_name": train_dataset.classes[target],
                "source_path": str(Path(source_path).resolve()),
                "relative_path": relative,
                "calibration_entropy": float(calibration["entropy"][index]),
                "holdout_entropy": float(holdout["entropy"][index]),
                "entropy_percentile": float(entropy_percentile[index]),
                "calibration_true_probability": float(calibration["true_probability"][index]),
                "holdout_true_probability": float(holdout["true_probability"][index]),
                "calibration_argmax_correct": float(calibration["argmax_correct"][index]),
                "holdout_argmax_correct": float(holdout["argmax_correct"][index]),
                "dino_own_centroid_similarity": float(own_similarity[index]),
                "dino_radial_cosine_distance": float(radial_distance[index]),
                "dino_radial_percentile": float(radial_percentile[index]),
            }
        )
    atomic_json(
        output_root / "preflight" / "per_image_teacher_entropy_and_dino.json",
        {
            "status": "complete",
            "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
            "teacher_checkpoint_sha256": file_sha256(args.teacher_checkpoint.resolve()),
            "images": rows,
        },
    )

    manifests = {}
    arm_summaries = {}
    selection_sets = {}
    targets = np.asarray(train_dataset.targets)
    for selection_seed in SELECTION_SEEDS:
        noise = np.asarray([gumbel_noise(selection_seed, row["relative_path"]) for row in rows])
        for lambda_value in LAMBDAS:
            selected_indices = []
            selected_records = []
            score = lambda_value * (2.0 * entropy_percentile - 1.0) + noise
            for class_id in range(CLASSES):
                class_indices = np.flatnonzero(targets == class_id)
                chosen = sorted(
                    class_indices,
                    key=lambda index: (-float(score[index]), rows[index]["relative_path"]),
                )[:IPC]
                if len(set(chosen)) != IPC:
                    raise RuntimeError("Gumbel Top-k produced duplicate images")
                selected_indices.extend(chosen)
                for slot, index in enumerate(chosen):
                    selected_records.append(
                        {
                            **rows[index],
                            "slot": slot,
                            "selection_score": float(score[index]),
                            "gumbel_noise": float(noise[index]),
                        }
                    )
            arm = f"lambda_{lambda_value:+d}_rseed{selection_seed}"
            manifest = {
                "status": "complete",
                "experiment": "imagenette_entropy_selection_v1",
                "dataset": "imagenet-nette",
                "classes": CLASSES,
                "ipc": IPC,
                "lambda": lambda_value,
                "selection_seed": selection_seed,
                "selection_method": "teacher_entropy_percentile_gumbel_topk",
                "selection_images": len(selected_records),
                "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
                "teacher_checkpoint_sha256": file_sha256(args.teacher_checkpoint.resolve()),
                "calibration_views": SELECTION_VIEWS,
                "calibration_view_seed": CALIBRATION_VIEW_SEED,
                "holdout_views": SELECTION_VIEWS,
                "holdout_view_seed": HOLDOUT_VIEW_SEED,
                "training_sample_weighting": "equal",
                "candidate_pool": "all images of the true class; no Teacher correctness/confidence filter",
                "candidate_images_per_class": {
                    class_name: len([target for target in train_dataset.targets if target == class_id])
                    for class_id, class_name in enumerate(train_dataset.classes)
                },
                "official_split_file_list": str(split_file_list_path),
                "official_split_file_list_sha256": split_file_list_sha256,
                "class_to_idx": train_dataset.class_to_idx,
                "images": selected_records,
            }
            manifest_path = output_root / "manifests" / f"{arm}.json"
            atomic_json(manifest_path, manifest)
            manifests[arm] = str(manifest_path)
            selected_set = {row["relative_path"] for row in selected_records}
            selection_sets[arm] = selected_set
            arm_summaries[arm] = selection_metrics(
                selected_indices, features, targets, rows
            )
            contact_sheet(
                output_root / "preflight" / "contact_sheets" / f"{arm}.jpg",
                selected_records,
                f"ImageNette IPC10 {arm}",
            )

    entropy_spread = []
    reproducibility = []
    for class_id in range(CLASSES):
        indices = np.flatnonzero(targets == class_id)
        values = calibration["entropy"][indices]
        entropy_spread.append(float(np.quantile(values, 0.95) - np.quantile(values, 0.05)))
        reproducibility.append(correlation(values, holdout["entropy"][indices]))
    all_calibration = calibration["entropy"]
    all_holdout = holdout["entropy"]
    overlap = {}
    for selection_seed in SELECTION_SEEDS:
        for left, right in zip(LAMBDAS[:-1], LAMBDAS[1:]):
            a = selection_sets[f"lambda_{left:+d}_rseed{selection_seed}"]
            b = selection_sets[f"lambda_{right:+d}_rseed{selection_seed}"]
            overlap[f"rseed{selection_seed}:{left:+d}_vs_{right:+d}"] = len(a & b) / len(a)
    preflight = {
        "status": "complete_pending_manual_visual_review",
        "experiment": "imagenette_entropy_selection_v1",
        "data_root": str(data_root),
        "train_images": len(train_dataset),
        "test_images": len(test_dataset),
        "classes": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "official_split_file_list": str(split_file_list_path),
        "official_split_file_list_sha256": split_file_list_sha256,
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": file_sha256(args.teacher_checkpoint.resolve()),
        "teacher_test": teacher_test,
        "teacher_mode": "eval_frozen_no_grad",
        "teacher_input": {
            "architecture": "torchvision ResNet18 with 10-class head",
            "normalization": "ImageNet mean/std",
            "class_mapping": "sorted official ImageFolder class folders",
        },
        "entropy": {
            "definition": "mean of 16 per-view normalized entropies; not entropy of mean probability",
            "logarithm": "natural",
            "computation_dtype": "float32 softmax/log_softmax; float64 accumulation only",
            "selection_uses": "calibration views only; holdout views are diagnostics only",
            "calibration_view_seed": CALIBRATION_VIEW_SEED,
            "holdout_view_seed": HOLDOUT_VIEW_SEED,
            "global_calibration_mean": float(all_calibration.mean()),
            "global_calibration_std": float(all_calibration.std(ddof=1)),
            "global_calibration_p05": float(np.quantile(all_calibration, 0.05)),
            "global_calibration_p95": float(np.quantile(all_calibration, 0.95)),
            "mean_class_p95_minus_p05": statistics.mean(entropy_spread),
            "global_calibration_holdout_pearson": correlation(all_calibration, all_holdout),
            "global_calibration_holdout_spearman": correlation(rankdata(all_calibration), rankdata(all_holdout)),
            "per_class_calibration_holdout_pearson": reproducibility,
        },
        "dino": {
            "cache": str(args.dino_cache.resolve()),
            "metadata": dino_metadata,
            "role": "companion audit only; not a selection condition",
        },
        "arm_summaries": arm_summaries,
        "adjacent_lambda_selection_overlap": overlap,
        "manifests": manifests,
        "manual_visual_review": {
            "required_before_training": True,
            "contact_sheet_directory": str(output_root / "preflight" / "contact_sheets"),
            "questions": [
                "subject missing or severe crop failure",
                "occlusion",
                "background dominance",
                "label anomaly",
            ],
            "approval_file": str(output_root / "preflight" / "manual_visual_gate.json"),
        },
    }
    atomic_json(summary_path, preflight)
    print(json.dumps({"status": preflight["status"], "manifests": len(manifests)}, indent=2))


if __name__ == "__main__":
    main()
