"""Relate holdout Teacher entropy to exact neighborhoods in original DINO space."""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, rankdata, spearmanr

from imagenette_entropy_protocol import atomic_json, file_sha256


KS = (10, 30, 50)


def exact_neighbors(features, targets, batch_size):
    device_features = features.cuda()
    n = len(features)
    global_neighbors = torch.empty(n, max(KS), dtype=torch.int64)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        similarity = device_features[start:stop] @ device_features.T
        rows = torch.arange(stop - start, device=similarity.device)
        similarity[rows, torch.arange(start, stop, device=similarity.device)] = -torch.inf
        global_neighbors[start:stop] = similarity.topk(max(KS), dim=1).indices.cpu()
        print(f"global neighbors={stop}/{n}", flush=True)
    within_distances = {k: torch.empty(n, dtype=torch.float32) for k in KS}
    for class_id in range(10):
        indices = torch.from_numpy(np.flatnonzero(targets == class_id)).long()
        class_features = device_features[indices.cuda()]
        similarity = class_features @ class_features.T
        similarity.fill_diagonal_(-torch.inf)
        nearest_similarity = similarity.topk(max(KS), dim=1).values.cpu()
        for k in KS:
            within_distances[k][indices] = (1.0 - nearest_similarity[:, :k]).mean(dim=1)
    return global_neighbors, within_distances


def correlation(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(pearsonr(x, y).statistic),
        "spearman": float(spearmanr(x, y).statistic),
    }


def summarize_class_values(values):
    raw = list(values)
    values = [float(value) for value in raw if value is not None and np.isfinite(value)]
    if not values:
        return {
            "mean": None,
            "sample_std": None,
            "positive_classes": 0,
            "zero_classes": 0,
            "negative_classes": 0,
            "undefined_classes": len(raw),
            "values": raw,
        }
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "positive_classes": sum(value > 0 for value in values),
        "zero_classes": sum(value == 0 for value in values),
        "negative_classes": sum(value < 0 for value in values),
        "undefined_classes": len(raw) - len(values),
        "values": raw,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-cache", required=True, type=Path)
    parser.add_argument("--per-image-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", default=1024, type=int)
    args = parser.parse_args()

    dino = torch.load(args.dino_cache, map_location="cpu", weights_only=False)
    split = dino["splits"]["train"]
    rows = json.loads(args.per_image_audit.read_text(encoding="utf-8"))["images"]
    if list(split["relative_paths"]) != [row["relative_path"] for row in rows]:
        raise RuntimeError("DINO and per-image audit order differs")
    targets = np.asarray(split["targets"], dtype=np.int64)
    features = F.normalize(split["features"].float(), dim=1)
    holdout_entropy = np.asarray([row["holdout_entropy"] for row in rows], dtype=np.float64)
    global_neighbors, within_distances = exact_neighbors(features, targets, args.batch_size)
    neighbor_targets = targets[global_neighbors.numpy()]

    metrics = {}
    for k in KS:
        interclass_ratio = (neighbor_targets[:, :k] != targets[:, None]).mean(axis=1)
        within_distance = within_distances[k].numpy().astype(np.float64)
        class_rows = []
        for class_id in range(10):
            indices = np.flatnonzero(targets == class_id)
            entropy_values = holdout_entropy[indices]
            low_threshold = np.quantile(entropy_values, 0.2)
            high_threshold = np.quantile(entropy_values, 0.8)
            low = indices[entropy_values <= low_threshold]
            high = indices[entropy_values >= high_threshold]
            class_rows.append(
                {
                    "class_id": class_id,
                    "images": len(indices),
                    "entropy_vs_interclass_neighbor_ratio": correlation(
                        holdout_entropy[indices], interclass_ratio[indices]
                    ),
                    "entropy_vs_within_class_neighbor_distance": correlation(
                        holdout_entropy[indices], within_distance[indices]
                    ),
                    "high_minus_low_entropy_quintile": {
                        "interclass_neighbor_ratio": float(
                            interclass_ratio[high].mean() - interclass_ratio[low].mean()
                        ),
                        "within_class_neighbor_distance": float(
                            within_distance[high].mean() - within_distance[low].mean()
                        ),
                        "high_images": len(high),
                        "low_images": len(low),
                    },
                }
            )
        entropy_within_class_rank = np.empty(len(targets), dtype=np.float64)
        interclass_within_class_rank = np.empty(len(targets), dtype=np.float64)
        distance_within_class_rank = np.empty(len(targets), dtype=np.float64)
        for class_id in range(10):
            indices = np.flatnonzero(targets == class_id)
            denominator = max(len(indices) - 1, 1)
            entropy_within_class_rank[indices] = (rankdata(holdout_entropy[indices], method="average") - 1) / denominator
            interclass_within_class_rank[indices] = (rankdata(interclass_ratio[indices], method="average") - 1) / denominator
            distance_within_class_rank[indices] = (rankdata(within_distance[indices], method="average") - 1) / denominator
        metrics[str(k)] = {
            "definitions": {
                "interclass_neighbor_ratio": f"fraction of the exact {k} nearest non-self DINO cosine neighbors with a different true class",
                "within_class_neighbor_distance": f"mean cosine distance to the exact {k} nearest non-self same-class DINO neighbors",
            },
            "per_class": class_rows,
            "class_summary": {
                "spearman_entropy_vs_interclass_neighbor_ratio": summarize_class_values(
                    row["entropy_vs_interclass_neighbor_ratio"]["spearman"] for row in class_rows
                ),
                "pearson_entropy_vs_interclass_neighbor_ratio": summarize_class_values(
                    row["entropy_vs_interclass_neighbor_ratio"]["pearson"] for row in class_rows
                ),
                "spearman_entropy_vs_within_class_neighbor_distance": summarize_class_values(
                    row["entropy_vs_within_class_neighbor_distance"]["spearman"] for row in class_rows
                ),
                "pearson_entropy_vs_within_class_neighbor_distance": summarize_class_values(
                    row["entropy_vs_within_class_neighbor_distance"]["pearson"] for row in class_rows
                ),
                "high_minus_low_quintile_interclass_neighbor_ratio": summarize_class_values(
                    row["high_minus_low_entropy_quintile"]["interclass_neighbor_ratio"] for row in class_rows
                ),
                "high_minus_low_quintile_within_class_neighbor_distance": summarize_class_values(
                    row["high_minus_low_entropy_quintile"]["within_class_neighbor_distance"] for row in class_rows
                ),
            },
            "pooled_within_class_rank_correlations": {
                "entropy_vs_interclass_neighbor_ratio": correlation(
                    entropy_within_class_rank, interclass_within_class_rank
                ),
                "entropy_vs_within_class_neighbor_distance": correlation(
                    entropy_within_class_rank, distance_within_class_rank
                ),
            },
            "per_image": {
                "interclass_neighbor_ratio": interclass_ratio.tolist(),
                "within_class_neighbor_distance": within_distance.tolist(),
            },
        }
    payload = {
        "status": "complete",
        "experiment": "imagenette_holdout_entropy_vs_original_dino_neighborhoods",
        "features": "frozen original DINO features, float32 L2-normalized; exact cosine neighbors; self excluded",
        "entropy": "mean normalized entropy over 16 independent holdout augmentation views",
        "class_control": "correlations computed separately within each true class, then summarized equally across 10 classes; pooled result uses within-class average ranks",
        "ks": list(KS),
        "metrics": metrics,
        "dino_cache": str(args.dino_cache.resolve()),
        "dino_cache_sha256": file_sha256(args.dino_cache.resolve()),
        "per_image_audit": str(args.per_image_audit.resolve()),
        "per_image_audit_sha256": file_sha256(args.per_image_audit.resolve()),
    }
    atomic_json(args.output, payload)
    print(json.dumps({"status": "complete", "images": len(targets), "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
