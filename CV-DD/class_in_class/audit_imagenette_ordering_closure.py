"""No-retraining closure audits for the ImageNette O/B/A/Aprime experiment.

The compute phase reconstructs the exact 2,000-epoch augmentation streams used by
each tuple, accumulates supervision matrices, and evaluates paired A/Aprime final
checkpoints.  Tuple caches make the expensive work restartable and shardable.
The template phase independently recomputes the 64 template views and compares
the first and last 32 views.  The assemble phase performs manifest-level blocked
inference and writes one authoritative JSON plus CSV matrices.
"""

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import kendalltau, pearsonr, spearmanr, t
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from imagenette_entropy_protocol import (
    CLASSES, DeterministicReleasedView, IMAGE_SIZE, TEST_IMAGES, TRAIN_EPOCHS, _rrc_params, _uniform, atomic_json,
    build_student, build_teacher, file_sha256, identity_seed, load_state_dict_payload,
    normalize, test_transform,
)
from prepare_imagenette_ordering_hierarchy import (
    MultiViewDataset, conditional_non_target, extract_probabilities, order_from_template,
)
from summarize_imagenette_ordering_hierarchy import PAIRS, block_adjusted


ARMS = ("O", "B", "A", "Aprime")
CLASS_NAMES = (
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
)


class HighTrainingViews(Dataset):
    def __init__(self, rows, student_seed):
        self.rows = rows
        self.student_seed = student_seed
        self.view = DeterministicReleasedView()
        self.base_tensors = []
        for row in rows:
            with Image.open(row["source_path"]) as image:
                image = image.convert("RGB")
                image = TF.resize(
                    image, IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR, antialias=True,
                )
                image = TF.center_crop(image, [IMAGE_SIZE, IMAGE_SIZE])
                self.base_tensors.append(TF.to_tensor(image))

    def __len__(self):
        return len(self.rows) * TRAIN_EPOCHS

    def __getitem__(self, task_index):
        epoch_index, image_index = divmod(task_index, len(self.rows))
        row = self.rows[image_index]
        seed = identity_seed(
            "imagenette-entropy-student-view-v1", self.student_seed,
            epoch_index + 1, row["relative_path"],
        )
        generator = torch.Generator().manual_seed(seed)
        tensor = self.base_tensors[image_index].clone()
        top, left, height, width = _rrc_params(tensor, generator)
        tensor = TF.resized_crop(
            tensor, top, left, height, width, [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=InterpolationMode.BILINEAR, antialias=True,
        )
        if float(torch.rand((), generator=generator)) < 0.5:
            tensor = TF.hflip(tensor)
        factors = {
            0: _uniform(generator, 0.6, 1.4),
            1: _uniform(generator, 0.6, 1.4),
            2: _uniform(generator, 0.6, 1.4),
        }
        for operation in torch.randperm(4, generator=generator).tolist():
            if operation == 0:
                tensor = TF.adjust_brightness(tensor, factors[0])
            elif operation == 1:
                tensor = TF.adjust_contrast(tensor, factors[1])
            elif operation == 2:
                tensor = TF.adjust_saturation(tensor, factors[2])
        alpha = torch.randn(3, generator=generator) * 0.1
        rgb = (
            self.view.lighting_eigenvectors * alpha.view(1, 3) * self.view.lighting_eigenvalues
        ).sum(dim=1)
        tensor = tensor + rgb.view(3, 1, 1)
        return tensor, int(row["class_id"]), image_index


def transformed(probabilities, targets, destinations, inverse=False):
    masked = probabilities.clone()
    masked.scatter_(1, targets[:, None], float("-inf"))
    sources = torch.argsort(masked, dim=1, descending=True, stable=True)[:, : CLASSES - 1]
    result = probabilities.clone()
    if inverse:
        result.scatter_(1, sources, probabilities.gather(1, destinations))
    else:
        result.scatter_(1, destinations, probabilities.gather(1, sources))
    return result


def high_rows_for_manifest(manifest):
    rows = sorted(manifest["images"], key=lambda row: (int(row["class_id"]), row["relative_path"]))
    high = []
    for class_id in range(CLASSES):
        group = sorted(
            (row for row in rows if int(row["class_id"]) == class_id),
            key=lambda row: (float(row["calibration_entropy"]), row["relative_path"]),
        )
        if len(group) != 10:
            raise RuntimeError("manifest is not class-balanced IPC10")
        high.extend(group[5:])
    return sorted(high, key=lambda row: (int(row["class_id"]), row["relative_path"]))


def destination_tables(rows, templates, manifest_seed):
    image = []
    klass = []
    for row in rows:
        target = int(row["class_id"])
        image.append(templates["image_templates"][row["relative_path"]]["full_order"])
        klass.append(templates["class_templates"][str(manifest_seed)][str(target)]["order"])
    return torch.tensor(image, dtype=torch.long), torch.tensor(klass, dtype=torch.long)


def evaluate_predictions(checkpoint, test_images, device, batch_size=512):
    model = build_student().to(device)
    model.load_state_dict(load_state_dict_payload(checkpoint), strict=True)
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(test_images), batch_size):
            images = test_images[start : start + batch_size].to(device, non_blocking=True)
            output.append(model(images).argmax(1).cpu())
    del model
    return torch.cat(output).numpy()


def compute_tuple(args, manifest_seed, student_seed, teacher, templates, test_images, test_targets):
    cache = args.output_root / "cache" / f"r{manifest_seed}_s{student_seed}.json"
    if cache.exists() and not args.force:
        print(f"reuse {cache}", flush=True)
        return
    manifest_path = args.experiment_root / "manifests" / f"lambda_+0_rseed{manifest_seed}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = high_rows_for_manifest(manifest)
    image_dest, class_dest = destination_tables(rows, templates, manifest_seed)
    dataset = HighTrainingViews(rows, student_seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        persistent_workers=args.workers > 0, pin_memory=True,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    sums = {arm: torch.zeros(CLASSES, CLASSES, dtype=torch.float64) for arm in ARMS}
    high_counts = torch.zeros(CLASSES, dtype=torch.int64)
    drift_sum = {arm: torch.zeros(CLASSES, CLASSES, dtype=torch.float64) for arm in ("A", "Aprime")}
    drift_sq = {arm: torch.zeros(CLASSES, dtype=torch.float64) for arm in ("A", "Aprime")}
    distance_max_difference = 0.0
    with torch.inference_mode():
        for batch_index, (images, targets, row_indices) in enumerate(loader):
            targets_gpu = targets.to(args.device, non_blocking=True)
            probabilities = F.softmax(teacher(normalize(images.to(args.device, non_blocking=True))).float(), dim=1)
            image_batch = image_dest[row_indices].to(args.device, non_blocking=True)
            class_batch = class_dest[row_indices].to(args.device, non_blocking=True)
            changed = {
                "O": probabilities,
                "B": transformed(probabilities, targets_gpu, image_batch),
                "A": transformed(probabilities, targets_gpu, class_batch),
                "Aprime": transformed(probabilities, targets_gpu, class_batch, inverse=True),
            }
            targets_cpu = targets.long()
            for arm, values in changed.items():
                sums[arm].index_add_(0, targets_cpu, values.double().cpu())
            high_counts.index_add_(0, targets_cpu, torch.ones_like(targets_cpu))
            for arm in ("A", "Aprime"):
                delta = changed[arm] - probabilities
                drift_sum[arm].index_add_(0, targets_cpu, delta.double().cpu())
                drift_sq[arm].index_add_(0, targets_cpu, delta.square().sum(1).double().cpu())
            d_a = (changed["A"] - probabilities).square().sum(1)
            d_p = (changed["Aprime"] - probabilities).square().sum(1)
            distance_max_difference = max(distance_max_difference, float((d_a - d_p).abs().max()))
            if batch_index % 100 == 0:
                print(
                    f"r{manifest_seed}s{student_seed} training views "
                    f"{min((batch_index + 1) * args.batch_size, len(dataset))}/{len(dataset)}",
                    flush=True,
                )
    expected_high = torch.full_like(high_counts, 5 * TRAIN_EPOCHS)
    if not torch.equal(high_counts, expected_high):
        raise RuntimeError("high-view class counts changed")
    total_counts = high_counts * 2
    for class_id in range(CLASSES):
        hard_low = float(5 * TRAIN_EPOCHS)
        for arm in ARMS:
            sums[arm][class_id, class_id] += hard_low
    matrices = {arm: (value / total_counts[:, None]).numpy() for arm, value in sums.items()}
    kappas = {}
    for arm in ("A", "Aprime"):
        numerator = drift_sum[arm].square().sum(1)
        denominator_high = high_counts.double() * drift_sq[arm]
        denominator_full = total_counts.double() * drift_sq[arm]
        kappas[arm] = {
            "intervened_high_views": torch.where(denominator_high > 0, numerator / denominator_high, 0).tolist(),
            "full_training_supervision": torch.where(denominator_full > 0, numerator / denominator_full, 0).tolist(),
            "mean_squared_l2_by_class": (drift_sq[arm] / high_counts).tolist(),
        }

    checkpoint_a = args.experiment_root / f"checkpoints/rseed{manifest_seed}/sseed{student_seed}/A/final.pth.tar"
    checkpoint_p = args.experiment_root / f"checkpoints/rseed{manifest_seed}/sseed{student_seed}/Aprime/final.pth.tar"
    pred_a = evaluate_predictions(checkpoint_a, test_images, args.device)
    pred_p = evaluate_predictions(checkpoint_p, test_images, args.device)
    confusion = {"A": np.zeros((CLASSES, CLASSES), dtype=np.int64), "Aprime": np.zeros((CLASSES, CLASSES), dtype=np.int64)}
    np.add.at(confusion["A"], (test_targets, pred_a), 1)
    np.add.at(confusion["Aprime"], (test_targets, pred_p), 1)
    ranks = {
        class_id: {value: rank + 1 for rank, value in enumerate(templates["class_templates"][str(manifest_seed)][str(class_id)]["order"])}
        for class_id in range(CLASSES)
    }
    rank_counts = {name: Counter() for name in ("A_all_errors", "Aprime_all_errors", "A_added", "Aprime_added")}
    a_added = (pred_a != test_targets) & (pred_p == test_targets)
    p_added = (pred_p != test_targets) & (pred_a == test_targets)
    for truth, pa, pp, add_a, add_p in zip(test_targets, pred_a, pred_p, a_added, p_added):
        if pa != truth:
            rank_counts["A_all_errors"][ranks[int(truth)][int(pa)]] += 1
        if pp != truth:
            rank_counts["Aprime_all_errors"][ranks[int(truth)][int(pp)]] += 1
        if add_a:
            rank_counts["A_added"][ranks[int(truth)][int(pa)]] += 1
        if add_p:
            rank_counts["Aprime_added"][ranks[int(truth)][int(pp)]] += 1
    result_payloads = {}
    for arm in ARMS:
        path = args.experiment_root / f"results/rseed{manifest_seed}/sseed{student_seed}/{arm}.json"
        result_payloads[arm] = json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "status": "complete", "manifest_seed": manifest_seed, "student_seed": student_seed,
        "manifest": str(manifest_path), "manifest_sha256": file_sha256(manifest_path),
        "training_high_views_per_class": high_counts.tolist(), "training_total_targets_per_class": total_counts.tolist(),
        "target_matrices": {arm: matrix.tolist() for arm, matrix in matrices.items()},
        "target_column_means": {arm: matrix.mean(0).tolist() for arm, matrix in matrices.items()},
        "kappa": kappas, "A_Aprime_squared_distance_max_difference": distance_max_difference,
        "drift_sums": {arm: drift_sum[arm].tolist() for arm in ("A", "Aprime")},
        "drift_squared_l2_sums": {arm: drift_sq[arm].tolist() for arm in ("A", "Aprime")},
        "test_confusion_counts": {arm: value.tolist() for arm, value in confusion.items()},
        "test_class_counts": np.bincount(test_targets, minlength=CLASSES).tolist(),
        "template_rank_error_counts": {key: [value.get(rank, 0) for rank in range(1, CLASSES)] for key, value in rank_counts.items()},
        "result_metrics": {arm: {"nll": value["final_loss"], "top1": value["final_top1"]} for arm, value in result_payloads.items()},
        "checkpoints": {"A": str(checkpoint_a), "Aprime": str(checkpoint_p)},
    }
    atomic_json(cache, payload)
    print(f"wrote {cache}", flush=True)


def rank_metrics(first_order, second_order, target):
    first = {value: rank for rank, value in enumerate(first_order)}
    second = {value: rank for rank, value in enumerate(second_order)}
    classes = [value for value in range(CLASSES) if value != target]
    x = [first[value] for value in classes]
    y = [second[value] for value in classes]
    return {
        "exact": first_order == second_order,
        "top1": first_order[0] == second_order[0],
        "spearman": float(spearmanr(x, y).statistic),
        "kendall": float(kendalltau(x, y).statistic),
    }


def distribution(values):
    values = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(values.mean()), "sample_std": float(values.std(ddof=1)),
        "min": float(values.min()), "median": float(np.median(values)), "max": float(values.max()),
    }


def compute_template_stability(args, teacher, templates):
    output = args.output_root / "template_half_stability.json"
    if output.exists() and not args.force:
        print(f"reuse {output}", flush=True)
        return
    paths = sorted(templates["image_templates"])
    row_by_path = {}
    for manifest_seed in PAIRS:
        manifest = json.loads((args.experiment_root / "manifests" / f"lambda_+0_rseed{manifest_seed}.json").read_text(encoding="utf-8"))
        row_by_path.update({row["relative_path"]: row for row in manifest["images"]})
    rows = [row_by_path[path] for path in paths]
    probabilities = extract_probabilities(
        teacher, rows, "imagenette-ordering-template-view-v1", 2026090703,
        64, args.batch_size, args.workers,
    )
    conditionals = {}
    image_metrics = []
    image_full_order_mismatches = 0
    for index, row in enumerate(rows):
        target = int(row["class_id"])
        conditional = conditional_non_target(probabilities[index], target)
        first = conditional[:32].mean(0)
        second = conditional[32:].mean(0)
        full = conditional.mean(0)
        first_order = order_from_template(first, target)
        second_order = order_from_template(second, target)
        image_full_order_mismatches += int(
            order_from_template(full, target)
            != templates["image_templates"][row["relative_path"]]["full_order"]
        )
        image_metrics.append(rank_metrics(first_order, second_order, target))
        conditionals[row["relative_path"]] = (first, second)
    class_metrics = []
    rows_by_manifest_class = []
    class_full_order_mismatches = 0
    for manifest_seed in PAIRS:
        for class_id in range(CLASSES):
            paths_here = templates["class_templates"][str(manifest_seed)][str(class_id)]["high_entropy_paths"]
            first = torch.stack([conditionals[path][0] for path in paths_here]).mean(0)
            second = torch.stack([conditionals[path][1] for path in paths_here]).mean(0)
            full = (first + second) / 2
            first_order = order_from_template(first, class_id)
            second_order = order_from_template(second, class_id)
            class_full_order_mismatches += int(
                order_from_template(full, class_id)
                != templates["class_templates"][str(manifest_seed)][str(class_id)]["order"]
            )
            metric = rank_metrics(first_order, second_order, class_id)
            class_metrics.append(metric)
            rows_by_manifest_class.append({"manifest_seed": manifest_seed, "class_id": class_id, **metric})
    def summarize(rows):
        return {
            "units": len(rows), "exact_order_rate": statistics.mean(row["exact"] for row in rows),
            "top1_same_rate": statistics.mean(row["top1"] for row in rows),
            "spearman": distribution(row["spearman"] for row in rows),
            "kendall_tau": distribution(row["kendall"] for row in rows),
        }
    atomic_json(output, {
        "status": "complete", "definition": "same frozen 64 template views split into first32 and last32",
        "B_image_template": summarize(image_metrics), "A_manifest_class_template": summarize(class_metrics),
        "recomputed_full_order_mismatches": {
            "B_image_template": image_full_order_mismatches,
            "A_manifest_class_template": class_full_order_mismatches,
        },
        "A_per_manifest_class": rows_by_manifest_class,
    })
    print(f"wrote {output}", flush=True)


def rank_summary(counts):
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    return {
        "count": total, "rank_counts_1_to_9": counts.tolist(),
        "top1_fraction": float(counts[0] / total) if total else 0.0,
        "top3_fraction": float(counts[:3].sum() / total) if total else 0.0,
        "mean_rank": float(np.dot(counts, np.arange(1, CLASSES)) / total) if total else 0.0,
    }


def assemble(args):
    caches = []
    for manifest_seed, students in PAIRS.items():
        for student_seed in students:
            path = args.output_root / "cache" / f"r{manifest_seed}_s{student_seed}.json"
            if not path.exists():
                raise RuntimeError(f"missing cache {path}")
            caches.append(json.loads(path.read_text(encoding="utf-8")))
    template_path = args.output_root / "template_half_stability.json"
    if not template_path.exists():
        raise RuntimeError(f"missing {template_path}")
    template_stability = json.loads(template_path.read_text(encoding="utf-8"))
    matrix_sums = {arm: np.zeros((CLASSES, CLASSES), dtype=np.float64) for arm in ARMS}
    drift_sums = {arm: np.zeros((CLASSES, CLASSES), dtype=np.float64) for arm in ("A", "Aprime")}
    drift_squared_l2_sums = {arm: np.zeros(CLASSES, dtype=np.float64) for arm in ("A", "Aprime")}
    high_counts = np.zeros(CLASSES, dtype=np.int64)
    total_counts = np.zeros(CLASSES, dtype=np.int64)
    denominator_differences = []
    confusions = {arm: np.zeros((CLASSES, CLASSES), dtype=np.int64) for arm in ("A", "Aprime")}
    test_counts = np.zeros(CLASSES, dtype=np.int64)
    rank_counts = {key: np.zeros(CLASSES - 1, dtype=np.int64) for key in ("A_all_errors", "Aprime_all_errors", "A_added", "Aprime_added")}
    for cache in caches:
        for arm in ARMS:
            matrix_sums[arm] += np.asarray(cache["target_matrices"][arm])
        high_counts += np.asarray(cache["training_high_views_per_class"], dtype=np.int64)
        total_counts += np.asarray(cache["training_total_targets_per_class"], dtype=np.int64)
        for arm in ("A", "Aprime"):
            drift_sums[arm] += np.asarray(cache["drift_sums"][arm], dtype=np.float64)
            drift_squared_l2_sums[arm] += np.asarray(cache["drift_squared_l2_sums"][arm], dtype=np.float64)
        a_sq = np.asarray(cache["kappa"]["A"]["mean_squared_l2_by_class"])
        p_sq = np.asarray(cache["kappa"]["Aprime"]["mean_squared_l2_by_class"])
        denominator_differences.append(float(np.max(np.abs(a_sq - p_sq))))
        for arm in confusions:
            confusions[arm] += np.asarray(cache["test_confusion_counts"][arm], dtype=np.int64)
        test_counts += np.asarray(cache["test_class_counts"], dtype=np.int64)
        for key in rank_counts:
            rank_counts[key] += np.asarray(cache["template_rank_error_counts"][key], dtype=np.int64)
    matrices = {arm: value / len(caches) for arm, value in matrix_sums.items()}
    column_means = {arm: value.mean(0) for arm, value in matrices.items()}
    kappa_summary = {}
    for arm in ("A", "Aprime"):
        numerator = np.square(drift_sums[arm]).sum(1)
        high = np.divide(
            numerator, high_counts * drift_squared_l2_sums[arm],
            out=np.zeros_like(numerator), where=drift_squared_l2_sums[arm] > 0,
        )
        full = np.divide(
            numerator, total_counts * drift_squared_l2_sums[arm],
            out=np.zeros_like(numerator), where=drift_squared_l2_sums[arm] > 0,
        )
        kappa_summary[arm] = {
            "intervened_high_views": {"per_class": high.tolist(), "class_mean": float(high.mean())},
            "full_training_supervision": {"per_class": full.tolist(), "class_mean": float(full.mean())},
        }
    confusion_rates = {arm: value / test_counts[:, None] for arm, value in confusions.items()}
    target_delta = matrices["A"] - matrices["Aprime"]
    confusion_delta = confusion_rates["A"] - confusion_rates["Aprime"]
    mask = ~np.eye(CLASSES, dtype=bool)
    relation = {
        "pearson_off_diagonal": {"r": float(pearsonr(target_delta[mask], confusion_delta[mask]).statistic), "p": float(pearsonr(target_delta[mask], confusion_delta[mask]).pvalue)},
        "spearman_off_diagonal": {"rho": float(spearmanr(target_delta[mask], confusion_delta[mask]).statistic), "p": float(spearmanr(target_delta[mask], confusion_delta[mask]).pvalue)},
    }
    summary = json.loads((args.experiment_root / "summary/ordering_hierarchy.json").read_text(encoding="utf-8"))
    manifest_rows = summary["manifest_rows"]
    for row in manifest_rows:
        row["contrasts"]["delta_Aprime_minus_O_nll"] = row["arm_nll"]["Aprime"] - row["arm_nll"]["O"]
        row["contrasts"]["delta_Aprime_minus_O_top1"] = row["arm_top1"]["Aprime"] - row["arm_top1"]["O"]
    aprime_o = {
        key: block_adjusted(manifest_rows, key)
        for key in ("delta_Aprime_minus_O_nll", "delta_Aprime_minus_O_top1")
    }
    output = {
        "status": "complete", "experiment": "imagenette_ordering_hierarchy_closure_audit_v1",
        "no_student_retraining": True, "tuple_caches": len(caches), "independent_units": 9,
        "class_names": list(CLASS_NAMES),
        "supervision": {
            "definition": "row=true class, column=class receiving supervision; exact 2000-epoch training views; includes high Soft and low Hard",
            "matrices": {arm: value.tolist() for arm, value in matrices.items()},
            "column_means": {arm: value.tolist() for arm, value in column_means.items()},
            "column_mean_delta_vs_O": {arm: (column_means[arm] - column_means["O"]).tolist() for arm in ARMS},
            "kappa": kappa_summary,
            "A_Aprime_squared_distance_denominator_max_difference": max(denominator_differences),
        },
        "test_error_flow": {
            "definition": "paired final checkpoints; confusion rows normalized by repeated test-class exposure across 18 tuples",
            "confusion_rates": {arm: value.tolist() for arm, value in confusion_rates.items()},
            "A_minus_Aprime_confusion_rate": confusion_delta.tolist(),
            "A_minus_Aprime_target_drift": target_delta.tolist(),
            "target_drift_vs_confusion_delta": relation,
            "template_rank_error_flow": {key: rank_summary(value) for key, value in rank_counts.items()},
        },
        "Aprime_vs_O_blocked_inference": aprime_o,
        "template_half_stability": template_stability,
        "sources": {
            "experiment_summary": str(args.experiment_root / "summary/ordering_hierarchy.json"),
            "experiment_summary_sha256": file_sha256(args.experiment_root / "summary/ordering_hierarchy.json"),
            "ordering_templates": str(args.experiment_root / "preflight/ordering_templates.json"),
            "ordering_templates_sha256": file_sha256(args.experiment_root / "preflight/ordering_templates.json"),
        },
    }
    output_path = args.output_root / "ordering_closure_audit.json"
    atomic_json(output_path, output)
    csv_dir = args.output_root / "matrices_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for arm, matrix in matrices.items():
        np.savetxt(csv_dir / f"supervision_{arm}.csv", matrix, delimiter=",")
    np.savetxt(csv_dir / "target_drift_A_minus_Aprime.csv", target_delta, delimiter=",")
    np.savetxt(csv_dir / "test_confusion_delta_A_minus_Aprime.csv", confusion_delta, delimiter=",")
    print(json.dumps({"status": "complete", "output": str(output_path), "tuple_caches": len(caches)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("compute", "templates", "assemble"), required=True)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest-seeds", default="3,4,5,6,7,8,9,10,11")
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--workers", default=10, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    # DataLoader workers provide the parallelism.  Prevent every worker and the
    # two audit shards from each creating a full OpenMP thread team.
    torch.set_num_threads(1)
    args.experiment_root = args.experiment_root.resolve()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    args.teacher_checkpoint = args.teacher_checkpoint.resolve()
    if args.phase == "assemble":
        assemble(args)
        return
    templates = json.loads((args.experiment_root / "preflight/ordering_templates.json").read_text(encoding="utf-8"))
    teacher = build_teacher(args.teacher_checkpoint).to(args.device).eval()
    if args.phase == "templates":
        compute_template_stability(args, teacher, templates)
        return
    test_dataset = datasets.ImageFolder(args.data_root / "test", transform=test_transform)
    if len(test_dataset) != TEST_IMAGES:
        raise RuntimeError("not the official ImageNette test split")
    print("caching deterministic test tensors", flush=True)
    test_images = []
    cache_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=args.workers,
        persistent_workers=False, pin_memory=False,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    for batch_index, (images, _) in enumerate(cache_loader):
        test_images.append(images)
        print(
            f"test tensors {min((batch_index + 1) * 256, len(test_dataset))}/{len(test_dataset)}",
            flush=True,
        )
    test_images = torch.cat(test_images).pin_memory()
    test_targets = np.asarray(test_dataset.targets, dtype=np.int64)
    for manifest_seed in map(int, args.manifest_seeds.split(",")):
        for student_seed in PAIRS[manifest_seed]:
            compute_tuple(args, manifest_seed, student_seed, teacher, templates, test_images, test_targets)


if __name__ == "__main__":
    main()
