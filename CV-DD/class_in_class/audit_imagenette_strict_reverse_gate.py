"""Exact-energy identifiability gate for a reverse class-template permutation.

No Student is trained.  For every exact training view, this reconstructs A and
searches all structurally exact-distance permutations for the one whose assigned
value ranks have minimum Spearman correlation with the class-template order.
The exhaustive 9! problem reduces exactly to choosing the orientation of each
length>=3 cycle in the view's rank permutation (at most 2^3 candidates).
"""

import argparse
import itertools
import json
import os
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from audit_imagenette_ordering_closure import HighTrainingViews, transformed
from imagenette_entropy_protocol import CLASSES, TRAIN_EPOCHS, atomic_json, build_teacher, file_sha256, normalize
from summarize_imagenette_ordering_hierarchy import PAIRS


ARMS = ("A", "Aprime", "Abar", "Abarprime")


def spearman_identity(permutation):
    n = len(permutation)
    return 1.0 - 6.0 * sum((index - value) ** 2 for index, value in enumerate(permutation)) / (n * (n * n - 1))


def exact_energy_candidates(sigma):
    """Return every generic-exact-energy output rank permutation.

    A's energy is the undirected edge multiset of the permutation sigma.  Every
    directed permutation with that edge multiset independently orients each
    cycle of length at least three; fixed points and 2-cycles have one orientation.
    """
    sigma = np.asarray(sigma, dtype=np.int64)
    inverse = np.argsort(sigma)
    seen = set()
    orientable = []
    for start in range(len(sigma)):
        if start in seen:
            continue
        cycle = []
        value = start
        while value not in seen:
            seen.add(value)
            cycle.append(value)
            value = int(sigma[value])
        if len(cycle) >= 3:
            orientable.append(cycle)
    candidates = set()
    for bits in itertools.product((0, 1), repeat=len(orientable)):
        rho = inverse.copy()
        for bit, cycle in zip(bits, orientable):
            if bit:
                for value in cycle:
                    rho[value] = sigma[value]
        candidates.add(tuple(int(value) for value in rho[sigma]))
    return sorted(candidates)


def best_reverse_permutation(sigma, cache):
    key = tuple(int(value) for value in sigma)
    if key not in cache:
        candidates = exact_energy_candidates(key)
        ranked = sorted((spearman_identity(candidate), candidate) for candidate in candidates)
        cache[key] = (ranked[0][1], ranked[0][0], len(candidates))
    return cache[key]


def strict_reverse_targets(probabilities, targets, destinations, cache):
    outputs = []
    inverse_outputs = []
    scores = []
    candidate_counts = []
    equals_a = []
    equals_aprime = []
    naive_reverse_energy_ratios = []
    probabilities_cpu = probabilities.detach().cpu()
    for probability, target_tensor, destination_tensor in zip(probabilities_cpu, targets.cpu(), destinations.cpu()):
        target = int(target_tensor)
        destination = [int(value) for value in destination_tensor]
        original_by_template = probability[destination].numpy()
        source_positions = np.argsort(-original_by_template, kind="stable")
        sorted_values = original_by_template[source_positions]
        sigma = np.argsort(source_positions)
        permutation, score, candidate_count = best_reverse_permutation(sigma, cache)
        source_classes = [destination[int(position)] for position in source_positions]
        mapping = list(range(CLASSES))
        for destination_position, source_rank in enumerate(permutation):
            mapping[destination[destination_position]] = source_classes[source_rank]
        inverse_mapping = np.argsort(mapping).tolist()
        abar = probability[torch.tensor(mapping, dtype=torch.long)]
        abarprime = probability[torch.tensor(inverse_mapping, dtype=torch.long)]
        a_mapping = list(range(CLASSES))
        for destination_position, source_rank in enumerate(range(CLASSES - 1)):
            a_mapping[destination[destination_position]] = source_classes[source_rank]
        a = probability[torch.tensor(a_mapping, dtype=torch.long)]
        aprime = probability[torch.tensor(np.argsort(a_mapping).tolist(), dtype=torch.long)]
        reverse = probability.clone()
        reverse[torch.tensor(destination)] = torch.tensor(sorted_values[::-1].copy())
        reference_energy = float((a - probability).square().sum())
        reverse_energy = float((reverse - probability).square().sum())
        outputs.append(abar)
        inverse_outputs.append(abarprime)
        scores.append(score)
        candidate_counts.append(candidate_count)
        equals_a.append(torch.equal(abar, a))
        equals_aprime.append(torch.equal(abar, aprime))
        naive_reverse_energy_ratios.append(reverse_energy / reference_energy if reference_energy > 0 else 1.0)
    return (
        torch.stack(outputs).to(probabilities.device),
        torch.stack(inverse_outputs).to(probabilities.device),
        np.asarray(scores), np.asarray(candidate_counts), np.asarray(equals_a),
        np.asarray(equals_aprime), np.asarray(naive_reverse_energy_ratios),
    )


def high_rows(manifest):
    rows = sorted(manifest["images"], key=lambda row: (int(row["class_id"]), row["relative_path"]))
    selected = []
    for class_id in range(CLASSES):
        group = sorted(
            (row for row in rows if int(row["class_id"]) == class_id),
            key=lambda row: (float(row["calibration_entropy"]), row["relative_path"]),
        )
        selected.extend(group[5:])
    return sorted(selected, key=lambda row: (int(row["class_id"]), row["relative_path"]))


def distribution(values, weights=None):
    values = np.asarray(values, dtype=np.float64)
    result = {
        "mean": float(values.mean()), "sample_std": float(values.std(ddof=1)),
        "min": float(values.min()), "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)), "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        result["energy_weighted_mean"] = float(np.dot(values, weights) / weights.sum())
    return result


def compute_tuple(args, manifest_seed, student_seed, teacher, templates):
    output = args.output_root / "cache" / f"r{manifest_seed}_s{student_seed}.json"
    if output.exists() and not args.force:
        print(f"reuse {output}", flush=True)
        return
    manifest_path = args.experiment_root / "manifests" / f"lambda_+0_rseed{manifest_seed}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = high_rows(manifest)
    destinations = torch.tensor([
        templates["class_templates"][str(manifest_seed)][str(int(row["class_id"]))]["order"]
        for row in rows
    ], dtype=torch.long)
    dataset = HighTrainingViews(rows, student_seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        persistent_workers=args.workers > 0, pin_memory=True,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    sums = {arm: torch.zeros(CLASSES, CLASSES, dtype=torch.float64) for arm in ARMS}
    squares = {arm: torch.zeros(CLASSES, dtype=torch.float64) for arm in ARMS}
    counts = torch.zeros(CLASSES, dtype=torch.int64)
    score_values = []
    energy_values = []
    candidate_histogram = Counter()
    equal_a = equal_aprime = 0
    equal_a_energy = equal_aprime_energy = 0.0
    naive_ratios = []
    maximum_energy_difference = 0.0
    maximum_multiset_difference = 0.0
    cache = {}
    with torch.inference_mode():
        for batch_index, (images, targets, row_indices) in enumerate(loader):
            targets_gpu = targets.to(args.device, non_blocking=True)
            original = F.softmax(teacher(normalize(images.to(args.device, non_blocking=True))).float(), dim=1)
            destination_batch = destinations[row_indices]
            a = transformed(original, targets_gpu, destination_batch.to(args.device))
            aprime = transformed(original, targets_gpu, destination_batch.to(args.device), inverse=True)
            abar, abarprime, scores, candidate_counts, equals_a, equals_aprime, ratios = strict_reverse_targets(
                original, targets_gpu, destination_batch, cache
            )
            changed = {"A": a, "Aprime": aprime, "Abar": abar, "Abarprime": abarprime}
            targets_cpu = targets.long()
            energies = {}
            for arm, values in changed.items():
                delta = values - original
                sums[arm].index_add_(0, targets_cpu, delta.double().cpu())
                energy = delta.square().sum(1)
                squares[arm].index_add_(0, targets_cpu, energy.double().cpu())
                energies[arm] = energy
                maximum_multiset_difference = max(
                    maximum_multiset_difference,
                    float((values.sort(1).values - original.sort(1).values).abs().max()),
                )
            maximum_energy_difference = max(
                maximum_energy_difference,
                float(torch.stack([(energies[arm] - energies["A"]).abs().max() for arm in ARMS]).max()),
            )
            counts.index_add_(0, targets_cpu, torch.ones_like(targets_cpu))
            energy_cpu = energies["A"].double().cpu().numpy()
            score_values.extend(scores.tolist())
            energy_values.extend(energy_cpu.tolist())
            candidate_histogram.update(map(int, candidate_counts))
            equal_a += int(equals_a.sum()); equal_aprime += int(equals_aprime.sum())
            equal_a_energy += float(energy_cpu[equals_a].sum())
            equal_aprime_energy += float(energy_cpu[equals_aprime].sum())
            naive_ratios.extend(ratios.tolist())
            if batch_index % 100 == 0:
                print(f"r{manifest_seed}s{student_seed} {min((batch_index+1)*args.batch_size,len(dataset))}/{len(dataset)}", flush=True)
    expected = torch.full_like(counts, 5 * TRAIN_EPOCHS)
    if not torch.equal(counts, expected):
        raise RuntimeError("training-view count mismatch")
    total_energy = float(sum(energy_values))
    payload = {
        "status": "complete", "manifest_seed": manifest_seed, "student_seed": student_seed,
        "views": len(dataset), "views_per_class": counts.tolist(),
        "drift_sums": {arm: value.tolist() for arm, value in sums.items()},
        "drift_squared_l2_sums": {arm: value.tolist() for arm, value in squares.items()},
        "maximum_per_view_energy_difference_vs_A": maximum_energy_difference,
        "maximum_probability_multiset_difference": maximum_multiset_difference,
        "achieved_template_spearman": distribution(score_values, energy_values),
        "candidate_count_histogram": {str(key): value for key, value in sorted(candidate_histogram.items())},
        "Abar_exact_overlap": {
            "with_A_view_fraction": equal_a / len(dataset),
            "with_Aprime_view_fraction": equal_aprime / len(dataset),
            "with_A_energy_fraction": equal_a_energy / total_energy,
            "with_Aprime_energy_fraction": equal_aprime_energy / total_energy,
        },
        "naive_reverse_energy_ratio_to_A": distribution(naive_ratios),
        "unique_rank_permutations": len(cache),
        "manifest": str(manifest_path), "manifest_sha256": file_sha256(manifest_path),
    }
    atomic_json(output, payload)
    print(f"wrote {output}", flush=True)


def cosine(left, right):
    left = np.asarray(left).reshape(-1); right = np.asarray(right).reshape(-1)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def assemble(args):
    payloads = []
    for manifest_seed, students in PAIRS.items():
        for student_seed in students:
            path = args.output_root / "cache" / f"r{manifest_seed}_s{student_seed}.json"
            if not path.exists():
                raise RuntimeError(f"missing {path}")
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
    sums = {arm: np.zeros((CLASSES, CLASSES)) for arm in ARMS}
    squares = {arm: np.zeros(CLASSES) for arm in ARMS}
    counts = np.zeros(CLASSES)
    for payload in payloads:
        counts += np.asarray(payload["views_per_class"])
        for arm in ARMS:
            sums[arm] += np.asarray(payload["drift_sums"][arm])
            squares[arm] += np.asarray(payload["drift_squared_l2_sums"][arm])
    means = {arm: sums[arm] / counts[:, None] for arm in ARMS}
    energy = {arm: squares[arm] / counts for arm in ARMS}
    kappa = {arm: np.square(means[arm]).sum(1) / energy[arm] for arm in ARMS}
    normalized = {arm: means[arm] / np.sqrt(energy["A"])[:, None] for arm in ARMS}
    reference = normalized["A"]
    geometry = {}
    for arm in ARMS:
        value = normalized[arm]
        geometry[arm] = {
            "macro_kappa": float(kappa[arm].mean()),
            "per_class_kappa": kappa[arm].tolist(),
            "cosine_to_A_mean_displacement": cosine(value, reference),
            "signed_projection_on_A_in_A_norm_units": float(np.sum(value * reference) / np.sum(reference * reference)),
            "orthogonal_norm_in_A_norm_units": float(np.linalg.norm(value - np.sum(value*reference)/np.sum(reference*reference)*reference) / np.linalg.norm(reference)),
        }
    total_views = sum(payload["views"] for payload in payloads)
    # Aggregate scalar diagnostics exactly from per-tuple fractions, with equal
    # tuple cardinality. Energy fractions are weighted by each tuple's A energy.
    energy_totals = [sum(payload["drift_squared_l2_sums"]["A"]) for payload in payloads]
    overlap = {}
    for key in ("with_A", "with_Aprime"):
        overlap[f"{key}_view_fraction"] = statistics.mean(payload["Abar_exact_overlap"][f"{key}_view_fraction"] for payload in payloads)
        overlap[f"{key}_energy_fraction"] = float(sum(
            energy_total * payload["Abar_exact_overlap"][f"{key}_energy_fraction"]
            for energy_total, payload in zip(energy_totals, payloads)
        ) / sum(energy_totals))
    candidate_counts = Counter()
    for payload in payloads:
        candidate_counts.update({int(key): value for key, value in payload["candidate_count_histogram"].items()})
    # Tuple distributions have equal cardinality; retain exact mean and extrema.
    score_means = [payload["achieved_template_spearman"]["mean"] for payload in payloads]
    score_weighted = [payload["achieved_template_spearman"]["energy_weighted_mean"] for payload in payloads]
    result = {
        "status": "complete_no_student_training", "experiment": "imagenette_strict_reverse_identifiability_gate_v1",
        "views": total_views, "tuples": len(payloads),
        "search": {
            "objective": "minimum Spearman correlation between assigned non-target value ranks and A class-template order",
            "constraint": "generic exact equality of each view's squared L2 displacement to A",
            "equivalence": "9! exhaustive search reduced to independent orientations of permutation cycles of length >=3",
            "tie_break": "lexicographically first permutation among equal minimum-Spearman candidates",
            "candidate_count_histogram": {str(key): value for key, value in sorted(candidate_counts.items())},
        },
        "geometry": geometry,
        "Abar_overlap": overlap,
        "achieved_template_spearman": {
            "tuple_mean_range": [min(score_means), max(score_means)],
            "mean": statistics.mean(score_means),
            "energy_weighted_mean": float(np.average(score_weighted, weights=energy_totals)),
        },
        "maximum_per_view_energy_difference_vs_A": max(payload["maximum_per_view_energy_difference_vs_A"] for payload in payloads),
        "maximum_probability_multiset_difference": max(payload["maximum_probability_multiset_difference"] for payload in payloads),
        "naive_reverse_energy_ratio_to_A": {
            "tuple_mean": statistics.mean(payload["naive_reverse_energy_ratio_to_A"]["mean"] for payload in payloads),
            "tuple_median_range": [min(payload["naive_reverse_energy_ratio_to_A"]["median"] for payload in payloads), max(payload["naive_reverse_energy_ratio_to_A"]["median"] for payload in payloads)],
        },
        "sources": {
            "templates": str(args.experiment_root / "preflight/ordering_templates.json"),
            "templates_sha256": file_sha256(args.experiment_root / "preflight/ordering_templates.json"),
        },
    }
    atomic_json(args.output_root / "strict_reverse_identifiability_gate.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("compute", "assemble"), required=True)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest-seeds", default="3,4,5,6,7,8,9,10,11")
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.experiment_root = args.experiment_root.resolve()
    args.teacher_checkpoint = args.teacher_checkpoint.resolve()
    args.output_root = args.output_root.resolve()
    if args.phase == "assemble":
        assemble(args)
        return
    templates = json.loads((args.experiment_root / "preflight/ordering_templates.json").read_text(encoding="utf-8"))
    teacher = build_teacher(args.teacher_checkpoint).to(args.device).eval()
    for manifest_seed in map(int, args.manifest_seeds.split(",")):
        for student_seed in PAIRS[manifest_seed]:
            compute_tuple(args, manifest_seed, student_seed, teacher, templates)


if __name__ == "__main__":
    main()
