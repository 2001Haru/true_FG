"""Plot shared-coordinate ImageNette class and Teacher-entropy t-SNE maps."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.neighbors import NearestNeighbors


SEMANTIC_NAMES = (
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
)
TSNE_SEEDS = (0, 1, 2)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def neighbor_indices(coordinates, neighbors=30):
    model = NearestNeighbors(n_neighbors=neighbors + 1, algorithm="kd_tree")
    return model.fit(coordinates).kneighbors(return_distance=False)[:, 1:]


def mean_neighbor_jaccard(left, right):
    values = []
    for a, b in zip(left, right):
        a, b = set(map(int, a)), set(map(int, b))
        values.append(len(a & b) / len(a | b))
    return float(np.mean(values))


def make_tsne(values, seed):
    return TSNE(
        n_components=2,
        perplexity=30.0,
        early_exaggeration=12.0,
        learning_rate="auto",
        max_iter=1500,
        init="pca",
        method="barnes_hut",
        angle=0.5,
        random_state=seed,
        verbose=1,
    ).fit_transform(values)


def shared_limits(coordinates):
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    padding = np.maximum((maximum - minimum) * 0.035, 1e-6)
    return (minimum - padding, maximum + padding)


def plot_pair(coordinates, targets, entropy, upper, output, title):
    figure, axes = plt.subplots(1, 2, figsize=(17, 7.8), sharex=True, sharey=True)
    limits = shared_limits(coordinates)
    categorical = plt.get_cmap("tab10")
    for class_id in range(10):
        mask = targets == class_id
        axes[0].scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=7,
            alpha=0.78,
            linewidths=0,
            color=categorical(class_id),
            rasterized=True,
        )
    entropy_order = np.argsort(entropy, kind="stable")
    axes[1].scatter(
        coordinates[entropy_order, 0],
        coordinates[entropy_order, 1],
        c=np.minimum(entropy[entropy_order], upper),
        cmap="viridis",
        norm=Normalize(0.0, upper),
        s=7,
        alpha=0.82,
        linewidths=0,
        rasterized=True,
    )
    for axis in axes:
        axis.set_xlim(limits[0][0], limits[1][0])
        axis.set_ylim(limits[0][1], limits[1][1])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("t-SNE 1")
        axis.set_ylabel("t-SNE 2")
        axis.grid(alpha=0.12, linewidth=0.5)
    axes[0].set_title("True class")
    axes[1].set_title("Teacher normalized entropy")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=categorical(class_id),
            markeredgewidth=0,
            markersize=6,
            label=f"{class_id}: {name}",
        )
        for class_id, name in enumerate(SEMANTIC_NAMES)
    ]
    axes[0].legend(
        handles=handles,
        loc="lower left",
        ncol=2,
        frameon=True,
        fontsize=8,
        markerscale=1.2,
    )
    scalar = matplotlib.cm.ScalarMappable(norm=Normalize(0.0, upper), cmap="viridis")
    colorbar = figure.colorbar(scalar, ax=axes[1], fraction=0.045, pad=0.025)
    if upper < 1.0:
        colorbar.set_label(f"H/log(10), clipped at global P99={upper:.3f}")
    else:
        colorbar.set_label("H/log(10), fixed full range [0,1]")
    figure.suptitle(title, fontsize=14)
    figure.subplots_adjust(left=0.055, right=0.95, bottom=0.08, top=0.90, wspace=0.10)
    figure.savefig(output.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-cache", required=True, type=Path)
    parser.add_argument("--prediction-cache", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dino = torch.load(args.dino_cache, map_location="cpu", weights_only=False)
    prediction = torch.load(args.prediction_cache, map_location="cpu", weights_only=False)
    split = dino["splits"]["train"]
    if list(split["relative_paths"]) != prediction["relative_paths"]:
        raise RuntimeError("DINO and Teacher image order differs")
    targets = np.asarray(prediction["targets"], dtype=np.int64)
    entropy = np.asarray(prediction["mean_view_normalized_entropy"], dtype=np.float64)
    features = np.asarray(split["features"].float(), dtype=np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)

    pca = PCA(n_components=50, svd_solver="randomized", iterated_power=7, random_state=0)
    pca50 = pca.fit_transform(features)
    coordinates = {}
    neighbor_graphs = {}
    for seed in TSNE_SEEDS:
        print(f"fitting t-SNE seed={seed}", flush=True)
        coordinates[seed] = make_tsne(pca50, seed)
        neighbor_graphs[seed] = neighbor_indices(coordinates[seed], neighbors=30)

    audit_rng = np.random.default_rng(2026090701)
    audit_indices = np.sort(
        np.concatenate(
            [
                audit_rng.choice(
                    np.flatnonzero(targets == class_id), size=200, replace=False
                )
                for class_id in range(10)
            ]
        )
    )
    stability = {}
    for seed in TSNE_SEEDS:
        stability[str(seed)] = {
            "trustworthiness_k10_stratified_subset2000": float(
                trustworthiness(
                    pca50[audit_indices],
                    coordinates[seed][audit_indices],
                    n_neighbors=10,
                )
            ),
            "trustworthiness_k30_stratified_subset2000": float(
                trustworthiness(
                    pca50[audit_indices],
                    coordinates[seed][audit_indices],
                    n_neighbors=30,
                )
            ),
        }
    pairwise = {}
    for left, right in ((0, 1), (0, 2), (1, 2)):
        pairwise[f"seed{left}_vs_seed{right}"] = {
            "mean_jaccard_of_30_nearest_neighbors": mean_neighbor_jaccard(
                neighbor_graphs[left], neighbor_graphs[right]
            )
        }

    p99 = float(np.quantile(entropy, 0.99))
    plot_pair(
        coordinates[0],
        targets,
        entropy,
        p99,
        args.output_dir / "D1_tsne_class_and_entropy_p99",
        "ImageNette DINO t-SNE: class structure and Teacher entropy (P99-clipped)",
    )
    plot_pair(
        coordinates[0],
        targets,
        entropy,
        1.0,
        args.output_dir / "D2_tsne_class_and_entropy_full_range",
        "ImageNette DINO t-SNE: class structure and Teacher entropy (full [0,1] scale)",
    )

    coordinate_path = args.output_dir / "tsne_coordinates.npz"
    np.savez_compressed(
        coordinate_path,
        pca50=pca50,
        tsne_seed0=coordinates[0],
        tsne_seed1=coordinates[1],
        tsne_seed2=coordinates[2],
        targets=targets,
        entropy=entropy,
        relative_paths=np.asarray(prediction["relative_paths"]),
        pca_components=pca.components_,
        pca_mean=pca.mean_,
        pca_explained_variance_ratio=pca.explained_variance_ratio_,
    )
    metadata = {
        "status": "complete",
        "images": len(features),
        "feature_preprocessing": "L2-normalized frozen DINO features followed by one global PCA50",
        "pca_components": 50,
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "tsne": {
            "main_seed": 0,
            "audit_seeds": list(TSNE_SEEDS),
            "perplexity": 30.0,
            "early_exaggeration": 12.0,
            "learning_rate": "auto",
            "max_iter": 1500,
            "init": "pca",
            "method": "barnes_hut",
            "angle": 0.5,
        },
        "entropy": {
            "definition": prediction["definition"]["color_entropy"],
            "global_p99": p99,
            "main_map_range": [0.0, p99],
            "main_map_clipping": "values above global P99 use the maximum color",
            "full_map_range": [0.0, 1.0],
        },
        "stability": stability,
        "pairwise_seed_stability": pairwise,
        "inputs": {
            "dino_cache": str(args.dino_cache.resolve()),
            "dino_cache_sha256": sha256(args.dino_cache.resolve()),
            "prediction_cache": str(args.prediction_cache.resolve()),
            "prediction_cache_sha256": sha256(args.prediction_cache.resolve()),
        },
    }
    metadata_path = args.output_dir / "tsne_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
