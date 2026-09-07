"""Separate concentrated-conflict and diffuse-insufficiency entropy sources without retraining."""

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from imagenette_entropy_protocol import (
    CLASSES,
    DeterministicReleasedView,
    HOLDOUT_VIEW_SEED,
    SELECTION_VIEWS,
    atomic_json,
    build_student,
    build_teacher,
    file_sha256,
    identity_seed,
    normalize,
    test_transform,
    validate_official_split,
)


CLASS_NAMES = (
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
)


class ShapeViewDataset(Dataset):
    def __init__(self, samples, data_root, namespace, phase_seed):
        self.samples = samples
        self.data_root = data_root
        self.namespace = namespace
        self.phase_seed = phase_seed
        self.view = DeterministicReleasedView()

    def __len__(self):
        return len(self.samples) * SELECTION_VIEWS

    def __getitem__(self, task_index):
        image_index, view_index = divmod(task_index, SELECTION_VIEWS)
        path, target = self.samples[image_index]
        relative = Path(path).relative_to(self.data_root).as_posix()
        seed = identity_seed(self.namespace, self.phase_seed, relative, view_index)
        with Image.open(path) as image:
            tensor = self.view(image, seed)
        return tensor, int(target), image_index


@torch.inference_mode()
def shape_statistics(teacher, samples, data_root, namespace, phase_seed, batch_size, workers):
    dataset = ShapeViewDataset(samples, data_root, namespace, phase_seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        persistent_workers=workers > 0, pin_memory=True,
        prefetch_factor=2 if workers > 0 else None,
    )
    keys = (
        "entropy", "true_probability", "maximum_probability",
        "non_target_effective_classes", "non_target_top1_share", "top2_total_mass",
    )
    sums = {key: torch.zeros(len(samples), dtype=torch.float64) for key in keys}
    counts = torch.zeros(len(samples), dtype=torch.int64)
    for batch_index, (images, targets, indices) in enumerate(loader):
        images = normalize(images.cuda(non_blocking=True))
        targets_gpu = targets.cuda(non_blocking=True)
        probabilities = F.softmax(teacher(images).float(), dim=1)
        entropy = -(probabilities * probabilities.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(1) / math.log(CLASSES)
        true_probability = probabilities.gather(1, targets_gpu[:, None]).squeeze(1)
        non_target = probabilities.clone()
        non_target.scatter_(1, targets_gpu[:, None], 0.0)
        non_target_mass = non_target.sum(1).clamp_min(torch.finfo(torch.float32).tiny)
        conditional = non_target / non_target_mass[:, None]
        conditional_log = conditional.clamp_min(torch.finfo(torch.float32).tiny).log()
        effective = (-(conditional * conditional_log).sum(1)).exp()
        non_target_top1 = conditional.max(1).values
        top2_total = true_probability + non_target.max(1).values
        values = {
            "entropy": entropy,
            "true_probability": true_probability,
            "maximum_probability": probabilities.max(1).values,
            "non_target_effective_classes": effective,
            "non_target_top1_share": non_target_top1,
            "top2_total_mass": top2_total,
        }
        indices = indices.long()
        for key, value in values.items():
            sums[key].index_add_(0, indices, value.double().cpu())
        counts.index_add_(0, indices, torch.ones_like(indices))
        if batch_index % 100 == 0:
            print(f"{namespace} views={min((batch_index+1)*batch_size,len(dataset))}/{len(dataset)}", flush=True)
    if not torch.equal(counts, torch.full_like(counts, SELECTION_VIEWS)):
        raise RuntimeError("shape audit did not receive exactly 16 views per image")
    return {key: (value / counts).numpy() for key, value in sums.items()}


def stratify(metrics, targets, paths):
    entropy = metrics["entropy"]
    effective = metrics["non_target_effective_classes"]
    top1_share = metrics["non_target_top1_share"]
    labels = np.full(len(targets), "not_high_entropy", dtype=object)
    sensitivity = np.full(len(targets), "not_high_entropy", dtype=object)
    per_class = []
    for class_id in range(CLASSES):
        indices = np.flatnonzero(targets == class_id)
        ranked_entropy = sorted(indices, key=lambda i: (entropy[i], paths[i]))
        high = ranked_entropy[-math.ceil(0.2 * len(indices)):]
        by_effective = sorted(high, key=lambda i: (effective[i], paths[i]))
        split = len(by_effective) // 2
        conflict = by_effective[:split]
        insufficient = by_effective[split:]
        labels[conflict] = "concentrated_conflict"
        labels[insufficient] = "diffuse_insufficiency"
        by_top1 = sorted(high, key=lambda i: (-top1_share[i], paths[i]))
        sensitivity[by_top1[:split]] = "concentrated_conflict"
        sensitivity[by_top1[split:]] = "diffuse_insufficiency"
        per_class.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "images": len(indices),
            "high_entropy_images": len(high),
            "conflict_images": len(conflict),
            "insufficiency_images": len(insufficient),
        })
    high_mask = labels != "not_high_entropy"
    overlap = float(np.mean(labels[high_mask] == sensitivity[high_mask]))
    return labels, sensitivity, per_class, overlap


def group_summary(metrics, labels, group):
    mask = labels == group
    return {
        "images": int(mask.sum()),
        **{
            key: {
                "mean": float(values[mask].mean()),
                "sample_std": float(values[mask].std(ddof=1)),
            }
            for key, values in metrics.items()
        },
    }


@torch.inference_mode()
def evaluate_checkpoint(path, loader):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_student()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.cuda().eval()
    probabilities = []
    predictions = []
    targets = []
    for images, batch_targets in loader:
        logits = model(images.cuda(non_blocking=True))
        probabilities.append(F.softmax(logits.float(), dim=1).cpu())
        predictions.append(logits.argmax(1).cpu())
        targets.append(batch_targets)
    del model
    torch.cuda.empty_cache()
    return torch.cat(probabilities).numpy(), torch.cat(predictions).numpy(), torch.cat(targets).numpy()


def readout(probabilities, predictions, targets, labels):
    output = {}
    for group in ("all", "concentrated_conflict", "diffuse_insufficiency"):
        mask = np.ones(len(targets), dtype=bool) if group == "all" else labels == group
        true_probability = probabilities[np.arange(len(targets)), targets]
        output[group] = {
            "images": int(mask.sum()),
            "accuracy": float((predictions[mask] == targets[mask]).mean() * 100.0),
            "nll": float(-np.log(np.clip(true_probability[mask], 1e-30, 1.0)).mean()),
        }
    return output


def stats(values):
    values = list(map(float, values))
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "positive": sum(value > 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "negative": sum(value < 0 for value in values),
        "values": values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--per-image-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--workers", default=16, type=int)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    root = args.experiment_root.resolve()
    train, test = validate_official_split(data_root)
    teacher = build_teacher(args.teacher_checkpoint.resolve()).cuda().eval()
    train_metrics = shape_statistics(
        teacher, train.samples, data_root,
        "imagenette-selection-holdout-view-v1", HOLDOUT_VIEW_SEED,
        args.batch_size, args.workers,
    )
    frozen_rows = json.loads(args.per_image_audit.read_text(encoding="utf-8"))["images"]
    max_entropy_delta = float(np.max(np.abs(train_metrics["entropy"] - np.asarray([row["holdout_entropy"] for row in frozen_rows]))))
    if max_entropy_delta > 2e-6:
        raise RuntimeError(f"holdout entropy differs from frozen audit: {max_entropy_delta}")
    test_metrics = shape_statistics(
        teacher, test.samples, data_root,
        "imagenette-entropy-source-test-view-v1", 2026090702,
        args.batch_size, args.workers,
    )
    del teacher
    torch.cuda.empty_cache()
    train_paths = [Path(path).relative_to(data_root).as_posix() for path, _ in train.samples]
    test_paths = [Path(path).relative_to(data_root).as_posix() for path, _ in test.samples]
    train_targets = np.asarray(train.targets)
    test_targets = np.asarray(test.targets)
    train_labels, train_sensitivity, train_class, train_overlap = stratify(train_metrics, train_targets, train_paths)
    test_labels, test_sensitivity, test_class, test_overlap = stratify(test_metrics, test_targets, test_paths)

    selection_composition = {}
    path_to_index = {path: index for index, path in enumerate(train_paths)}
    for lambda_value in (-4, -2, 0, 2, 4, 8, 16, 32):
        rows = []
        for selection_seed in (0, 1, 2):
            manifest = json.loads((root / "manifests" / f"lambda_{lambda_value:+d}_rseed{selection_seed}.json").read_text(encoding="utf-8"))
            indices = [path_to_index[row["relative_path"]] for row in manifest["images"]]
            rows.append({
                "selection_seed": selection_seed,
                "mean_entropy": float(train_metrics["entropy"][indices].mean()),
                "mean_non_target_effective_classes": float(train_metrics["non_target_effective_classes"][indices].mean()),
                "concentrated_conflict": int(np.sum(train_labels[indices] == "concentrated_conflict")),
                "diffuse_insufficiency": int(np.sum(train_labels[indices] == "diffuse_insufficiency")),
            })
        selection_composition[str(lambda_value)] = rows
    top_manifest = json.loads((root / "manifests/highest_entropy_top10_per_class.json").read_text(encoding="utf-8"))
    top_indices = [path_to_index[row["relative_path"]] for row in top_manifest["images"]]
    selection_composition["highest_entropy_top10"] = [{
        "selection_seed": None,
        "mean_entropy": float(train_metrics["entropy"][top_indices].mean()),
        "mean_non_target_effective_classes": float(train_metrics["non_target_effective_classes"][top_indices].mean()),
        "concentrated_conflict": int(np.sum(train_labels[top_indices] == "concentrated_conflict")),
        "diffuse_insufficiency": int(np.sum(train_labels[top_indices] == "diffuse_insufficiency")),
    }]

    test_loader = DataLoader(
        datasets.ImageFolder(data_root / "test", transform=test_transform),
        batch_size=256, shuffle=False, num_workers=args.workers,
        persistent_workers=args.workers > 0, pin_memory=True,
    )
    tuple_rows = []
    summaries = {group: {metric: [] for metric in (
        "accuracy_identity_value_conflict", "accuracy_identity_value_insufficiency",
        "accuracy_conflict_minus_insufficiency", "nll_identity_value_conflict",
        "nll_identity_value_insufficiency", "nll_conflict_minus_insufficiency",
    )} for group in ("high", "low")}
    for group in ("high", "low"):
        for selection_seed in (0, 1, 2):
            for student_seed in (42, 43, 44):
                original_result = json.loads((root / "results/lambda0_entropy_allocation" / f"rseed{selection_seed}" / f"{group}_entropy_soft1_sseed{student_seed}.json").read_text(encoding="utf-8"))
                permuted_result = json.loads((root / "results/lambda0_nontarget_permutation" / f"rseed{selection_seed}" / f"{group}_entropy_permuted_soft1_sseed{student_seed}.json").read_text(encoding="utf-8"))
                original_prob, original_pred, loaded_targets = evaluate_checkpoint(Path(original_result["final_checkpoint"]), test_loader)
                permuted_prob, permuted_pred, loaded_targets_2 = evaluate_checkpoint(Path(permuted_result["final_checkpoint"]), test_loader)
                if not np.array_equal(loaded_targets, test_targets) or not np.array_equal(loaded_targets_2, test_targets):
                    raise RuntimeError("test order changed during checkpoint readout")
                original = readout(original_prob, original_pred, test_targets, test_labels)
                permuted = readout(permuted_prob, permuted_pred, test_targets, test_labels)
                row = {"group": group, "selection_seed": selection_seed, "student_seed": student_seed, "original": original, "permuted": permuted}
                for stratum, suffix in (("concentrated_conflict", "conflict"), ("diffuse_insufficiency", "insufficiency")):
                    row[f"accuracy_identity_value_{suffix}"] = original[stratum]["accuracy"] - permuted[stratum]["accuracy"]
                    row[f"nll_identity_value_{suffix}"] = permuted[stratum]["nll"] - original[stratum]["nll"]
                row["accuracy_conflict_minus_insufficiency"] = row["accuracy_identity_value_conflict"] - row["accuracy_identity_value_insufficiency"]
                row["nll_conflict_minus_insufficiency"] = row["nll_identity_value_conflict"] - row["nll_identity_value_insufficiency"]
                tuple_rows.append(row)
                for key in summaries[group]:
                    summaries[group][key].append(row[key])
                print(f"readout group={group} selection={selection_seed} student={student_seed}", flush=True)

    payload = {
        "status": "complete",
        "experiment": "imagenette_entropy_source_shape_stratified_checkpoint_readout",
        "no_student_retraining": True,
        "shape_definition": {
            "non_target_distribution": "q_k/(1-q_y) over k != y for each Teacher view",
            "effective_classes": "exp(-sum r_k log r_k), range [1,9], averaged across 16 views",
            "non_target_top1_share": "max r_k, averaged across 16 views",
            "top2_total_mass": "q_y + max_{k!=y} q_k, averaged across 16 views",
            "entropy": "mean per-view normalized entropy H(q)/log(10)",
        },
        "stratification": "within each true class take top 20% entropy; lower half effective classes = concentrated conflict, upper half = diffuse insufficiency; deterministic path tie-break",
        "sensitivity": "same high-entropy pool split by descending non-target top1 share",
        "train_shape_groups": {
            "per_class": train_class,
            "effective_vs_top1_label_agreement": train_overlap,
            "concentrated_conflict": group_summary(train_metrics, train_labels, "concentrated_conflict"),
            "diffuse_insufficiency": group_summary(train_metrics, train_labels, "diffuse_insufficiency"),
        },
        "test_shape_groups": {
            "per_class": test_class,
            "effective_vs_top1_label_agreement": test_overlap,
            "concentrated_conflict": group_summary(test_metrics, test_labels, "concentrated_conflict"),
            "diffuse_insufficiency": group_summary(test_metrics, test_labels, "diffuse_insufficiency"),
        },
        "selection_composition": selection_composition,
        "checkpoint_readout": {
            "metric_sign": "positive identity value means original labels outperform permuted labels",
            "summary": {group: {key: stats(values) for key, values in group_values.items()} for group, group_values in summaries.items()},
            "tuples": tuple_rows,
        },
        "train_per_image": {
            "relative_paths": train_paths,
            "targets": train_targets.tolist(),
            "stratum": train_labels.tolist(),
            "top1_sensitivity_stratum": train_sensitivity.tolist(),
            **{key: values.tolist() for key, values in train_metrics.items()},
        },
        "test_per_image": {
            "relative_paths": test_paths,
            "targets": test_targets.tolist(),
            "stratum": test_labels.tolist(),
            "top1_sensitivity_stratum": test_sensitivity.tolist(),
            **{key: values.tolist() for key, values in test_metrics.items()},
        },
        "holdout_entropy_max_abs_delta_vs_frozen": max_entropy_delta,
        "teacher_checkpoint_sha256": file_sha256(args.teacher_checkpoint.resolve()),
        "per_image_audit_sha256": file_sha256(args.per_image_audit.resolve()),
    }
    atomic_json(args.output, payload)
    print(json.dumps({"status": "complete", "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
