"""Generate publication-ready ImageNette entropy maps, heatmap, and image strips."""

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
from PIL import Image, ImageOps
from sklearn.decomposition import PCA


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


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure(fig, stem, dpi=220):
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def scatter_panels(coordinates, entropy, correct, targets, output, title, axis_labels, panel_notes=None):
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.6), sharex=panel_notes is None, sharey=panel_notes is None)
    norm = Normalize(0.0, 1.0)
    cmap = plt.get_cmap("viridis")
    for class_id, axis in enumerate(axes.flat):
        mask = targets == class_id
        ok = mask & correct
        wrong = mask & ~correct
        axis.scatter(
            coordinates[ok, 0], coordinates[ok, 1], c=entropy[ok], cmap=cmap,
            norm=norm, s=8, alpha=0.50, linewidths=0, rasterized=True,
        )
        axis.scatter(
            coordinates[wrong, 0], coordinates[wrong, 1], c=entropy[wrong], cmap=cmap,
            norm=norm, s=22, alpha=0.65, edgecolors="#d62728", linewidths=0.9,
            rasterized=True,
        )
        note = "" if panel_notes is None else f"\nEV {panel_notes[class_id][0]:.1f}% + {panel_notes[class_id][1]:.1f}%"
        axis.set_title(f"{class_id}: {SEMANTIC_NAMES[class_id]}{note}", fontsize=10)
        axis.tick_params(labelsize=7)
        axis.grid(alpha=0.18, linewidth=0.5)
        if class_id // 5 == 1:
            axis.set_xlabel(axis_labels[0], fontsize=9)
        if class_id % 5 == 0:
            axis.set_ylabel(axis_labels[1], fontsize=9)
    scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.subplots_adjust(bottom=0.10, top=0.90, left=0.055, right=0.91, wspace=0.20, hspace=0.32)
    colorbar_axis = fig.add_axes([0.925, 0.20, 0.012, 0.58])
    colorbar = fig.colorbar(scalar, cax=colorbar_axis)
    colorbar.set_label("Teacher normalized entropy H / log(10)", fontsize=9)
    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.45", markeredgewidth=0, markersize=5, label="All 16 calibration views correct"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.45", markeredgecolor="#d62728", markeredgewidth=1.2, markersize=7, label="At least one calibration view wrong"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=14, y=1.01)
    save_figure(fig, output)


def entropy_heatmap(entropy, targets, output):
    bins = np.linspace(0.0, 1.0, 21)
    matrix = np.zeros((10, 20), dtype=np.float64)
    medians = np.zeros(10)
    for class_id in range(10):
        values = entropy[targets == class_id]
        matrix[class_id] = np.histogram(values, bins=bins)[0] / len(values)
        medians[class_id] = np.median(values)
    fig, axis = plt.subplots(figsize=(14, 5.4))
    image = axis.imshow(matrix, aspect="auto", cmap="magma", vmin=0, vmax=matrix.max())
    axis.set_yticks(range(10), [f"{i}: {name}" for i, name in enumerate(SEMANTIC_NAMES)])
    tick_positions = np.arange(0, 21, 2) - 0.5
    axis.set_xticks(tick_positions, [f"{value:.1f}" for value in np.linspace(0, 1, 11)])
    axis.set_xlabel("Teacher normalized entropy bin (width 0.05)")
    axis.set_ylabel("True class")
    for class_id, median in enumerate(medians):
        x = median * 20 - 0.5
        axis.plot([x, x], [class_id - 0.44, class_id + 0.44], color="white", linewidth=2.5)
        axis.plot([x, x], [class_id - 0.44, class_id + 0.44], color="black", linewidth=0.8)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Within-class image proportion")
    axis.set_title("ImageNette class × Teacher entropy distribution\nshort black/white marker = class median")
    fig.tight_layout()
    save_figure(fig, output)
    return matrix, medians


def choose_representatives(indices, entropy, correct):
    ordered = sorted(indices, key=lambda index: (entropy[index], index))
    median = float(np.median(entropy[indices]))
    errors = sorted(
        [index for index in indices if not correct[index]],
        key=lambda index: (-entropy[index], index),
    )
    return {
        "lowest entropy": ordered[:5],
        "near median": sorted(indices, key=lambda index: (abs(entropy[index] - median), index))[:5],
        "highest entropy": ordered[-5:][::-1],
        "highest-entropy errors": errors[:5],
    }


def load_thumbnail(path, size=(150, 110)):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, "white")
        canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return np.asarray(canvas)


def representative_strips(
    source_paths,
    entropy,
    predictions,
    maximum,
    correct,
    view_correct_rate,
    wrong_predictions,
    wrong_maximum,
    targets,
    output_dir,
):
    selections = {}
    for class_id in range(10):
        indices = np.flatnonzero(targets == class_id).tolist()
        selections[class_id] = choose_representatives(indices, entropy, correct)

    fig, axes = plt.subplots(10, 20, figsize=(34, 18))
    group_names = tuple(next(iter(selections.values())).keys())
    for class_id in range(10):
        column = 0
        for group_name in group_names:
            chosen = selections[class_id][group_name]
            for slot in range(5):
                axis = axes[class_id, column]
                axis.axis("off")
                if slot < len(chosen):
                    index = chosen[slot]
                    axis.imshow(load_thumbnail(source_paths[index]))
                    is_error_strip = group_name == "highest-entropy errors"
                    shown_prediction = wrong_predictions[index] if is_error_strip else predictions[index]
                    shown_maximum = wrong_maximum[index] if is_error_strip else maximum[index]
                    correct_views = int(round(view_correct_rate[index] * 16))
                    state = f"views={correct_views}/16"
                    axis.set_title(
                        f"H={entropy[index]:.3f} p={shown_maximum:.2f}\npred={shown_prediction} {state}",
                        fontsize=6,
                        color="#b2182b" if correct_views < 16 else "black",
                    )
                if class_id == 0 and slot == 2:
                    axis.text(0.5, 1.34, group_name, transform=axis.transAxes, ha="center", va="bottom", fontsize=10, fontweight="bold")
                column += 1
        axes[class_id, 0].text(
            -0.18, 0.5, f"{class_id}\n{SEMANTIC_NAMES[class_id]}",
            transform=axes[class_id, 0].transAxes, ha="right", va="center", fontsize=9,
        )
    fig.suptitle("ImageNette representative images by Teacher entropy", fontsize=15, y=0.995)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.965, bottom=0.015, wspace=0.04, hspace=0.38)
    save_figure(fig, output_dir / "C_representative_strips_overview", dpi=180)

    for class_id in range(10):
        fig, axes = plt.subplots(4, 5, figsize=(12, 9))
        for row_id, group_name in enumerate(group_names):
            chosen = selections[class_id][group_name]
            for slot, axis in enumerate(axes[row_id]):
                axis.axis("off")
                if slot < len(chosen):
                    index = chosen[slot]
                    axis.imshow(load_thumbnail(source_paths[index], (210, 150)))
                    is_error_strip = group_name == "highest-entropy errors"
                    shown_prediction = wrong_predictions[index] if is_error_strip else predictions[index]
                    shown_maximum = wrong_maximum[index] if is_error_strip else maximum[index]
                    correct_views = int(round(view_correct_rate[index] * 16))
                    axis.set_title(
                        f"H={entropy[index]:.3f}  p={shown_maximum:.3f}\n"
                        f"pred={shown_prediction}  correct={correct_views}/16",
                        fontsize=8,
                        color="#b2182b" if correct_views < 16 else "black",
                    )
                if slot == 0:
                    axis.text(-0.10, 0.5, group_name, transform=axis.transAxes, ha="right", va="center", fontsize=9)
        fig.suptitle(f"Class {class_id}: {SEMANTIC_NAMES[class_id]}", fontsize=14)
        fig.subplots_adjust(left=0.13, right=0.99, top=0.93, bottom=0.03, wspace=0.08, hspace=0.40)
        save_figure(fig, output_dir / f"C_class_{class_id:02d}_{SEMANTIC_NAMES[class_id].replace(' ', '_')}", dpi=180)
    return selections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dino-cache", required=True, type=Path)
    parser.add_argument("--prediction-cache", required=True, type=Path)
    parser.add_argument("--per-image-audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dino = torch.load(args.dino_cache, map_location="cpu", weights_only=False)
    prediction = torch.load(args.prediction_cache, map_location="cpu", weights_only=False)
    per_image = json.loads(args.per_image_audit.read_text(encoding="utf-8"))["images"]
    split = dino["splits"]["train"]
    paths = list(split["relative_paths"])
    if paths != prediction["relative_paths"] or paths != [row["relative_path"] for row in per_image]:
        raise RuntimeError("DINO, Teacher, and per-image paths are not aligned")
    targets = np.asarray(prediction["targets"], dtype=np.int64)
    if not np.array_equal(targets, np.asarray(split["targets"], dtype=np.int64)):
        raise RuntimeError("DINO and Teacher targets differ")
    features = np.asarray(split["features"].float(), dtype=np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    entropy = np.asarray(prediction["mean_view_normalized_entropy"], dtype=np.float64)
    predictions = np.asarray(prediction["aggregate_prediction"], dtype=np.int64)
    maximum = np.asarray(prediction["aggregate_maximum_probability"], dtype=np.float64)
    view_correct_rate = np.asarray(prediction["view_correct_rate"], dtype=np.float64)
    correct = np.asarray(prediction["all_views_correct"], dtype=bool)
    wrong_predictions = np.asarray(prediction["highest_wrong_view_prediction"], dtype=np.int64)
    wrong_maximum = np.asarray(prediction["highest_wrong_view_maximum_probability"], dtype=np.float64)
    source_paths = [row["source_path"] for row in per_image]

    global_pca = PCA(n_components=2, svd_solver="full")
    global_coordinates = global_pca.fit_transform(features)
    global_ev = global_pca.explained_variance_ratio_ * 100
    scatter_panels(
        global_coordinates,
        entropy,
        correct,
        targets,
        args.output_dir / "A1_global_dino_pca_entropy_map",
        "ImageNette Teacher entropy on a shared global DINO PCA space",
        (f"Global PC1 ({global_ev[0]:.2f}% variance)", f"Global PC2 ({global_ev[1]:.2f}% variance)"),
    )

    local_coordinates = np.zeros((len(features), 2), dtype=np.float64)
    local_ev = []
    local_models = {}
    for class_id in range(10):
        indices = np.flatnonzero(targets == class_id)
        model = PCA(n_components=2, svd_solver="randomized", iterated_power=7, random_state=0)
        local_coordinates[indices] = model.fit_transform(features[indices])
        local_ev.append(model.explained_variance_ratio_ * 100)
        local_models[str(class_id)] = {
            "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
            "components": model.components_.tolist(),
            "mean": model.mean_.tolist(),
        }
    scatter_panels(
        local_coordinates,
        entropy,
        correct,
        targets,
        args.output_dir / "A2_per_class_dino_pca_entropy_map",
        "ImageNette Teacher entropy in per-class DINO PCA spaces",
        ("Class-specific PC1", "Class-specific PC2"),
        panel_notes=local_ev,
    )
    heatmap, medians = entropy_heatmap(
        entropy, targets, args.output_dir / "B_class_entropy_histogram_heatmap"
    )
    selections = representative_strips(
        source_paths,
        entropy,
        predictions,
        maximum,
        correct,
        view_correct_rate,
        wrong_predictions,
        wrong_maximum,
        targets,
        args.output_dir,
    )

    np.savez_compressed(
        args.output_dir / "pca_coordinates_and_entropy.npz",
        global_coordinates=global_coordinates,
        local_coordinates=local_coordinates,
        entropy=entropy,
        targets=targets,
        predictions=predictions,
        maximum_probability=maximum,
        correct=correct,
        global_components=global_pca.components_,
        global_mean=global_pca.mean_,
        global_explained_variance_ratio=global_pca.explained_variance_ratio_,
        class_entropy_histogram=heatmap,
        class_entropy_median=medians,
    )
    metadata = {
        "status": "complete",
        "images": len(features),
        "class_names": list(SEMANTIC_NAMES),
        "coordinate_definition": {
            "global": "sklearn exact full PCA fitted once on all L2-normalized DINO features",
            "per_class": "sklearn deterministic randomized PCA fitted separately per true class; directions/coordinates are not comparable across panels",
        },
        "entropy_definition": prediction["definition"]["color_entropy"],
        "prediction_definition": prediction["definition"]["image_prediction"],
        "global_explained_variance_ratio": global_pca.explained_variance_ratio_.tolist(),
        "per_class_pca": local_models,
        "all_16_views_correct_fraction": float(correct.mean()),
        "mean_per_view_teacher_accuracy": float(view_correct_rate.mean()),
        "aggregate_mean_probability_teacher_accuracy": float(np.asarray(prediction["aggregate_correct"], dtype=bool).mean()),
        "images_with_any_wrong_view_per_class": [int((~correct[targets == class_id]).sum()) for class_id in range(10)],
        "entropy_median_per_class": medians.tolist(),
        "representatives": {
            str(class_id): {
                group: [paths[index] for index in indices]
                for group, indices in groups.items()
            }
            for class_id, groups in selections.items()
        },
        "inputs": {
            "dino_cache": str(args.dino_cache.resolve()),
            "dino_cache_sha256": sha256(args.dino_cache.resolve()),
            "prediction_cache": str(args.prediction_cache.resolve()),
            "prediction_cache_sha256": sha256(args.prediction_cache.resolve()),
            "per_image_audit": str(args.per_image_audit.resolve()),
            "per_image_audit_sha256": sha256(args.per_image_audit.resolve()),
        },
        "outputs": {},
    }
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "figure_metadata.json":
            metadata["outputs"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (args.output_dir / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output_dir": str(args.output_dir.resolve()), "files": len(metadata["outputs"])}, indent=2))


if __name__ == "__main__":
    main()
