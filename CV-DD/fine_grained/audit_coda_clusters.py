"""Audit CoDA centers and preserve per-source-image clustering provenance."""

import argparse
import hashlib
import json
import math
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
FINAL_ORIGINS = {"hdbscan_initial", "hdbscan_kmeans_split", "kmeans_outliers"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"Invalid JSONL {path}:{line_number}") from error
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--classes", required=True, type=int)
    parser.add_argument("--ipc", required=True, type=int)
    parser.add_argument("--feature-space", required=True, choices=("vae", "dinov2"))
    parser.add_argument("--n-neighbors", required=True, type=int)
    parser.add_argument("--min-cluster-size", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    chunks = math.ceil(args.classes / 10)
    combined = {}
    cluster_hashes = {}
    provenance_hashes = {}
    rows = []
    for chunk_id in range(chunks):
        cluster_path = args.cluster_dir / (
            f"{args.ipc}_n_{args.n_neighbors}_s_{args.min_cluster_size}_saved_clusters_{chunk_id}.pkl"
        )
        provenance_path = args.cluster_dir / (
            f"{args.ipc}_n_{args.n_neighbors}_s_{args.min_cluster_size}_image_provenance_{chunk_id}.jsonl"
        )
        for path in (cluster_path, provenance_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        with cluster_path.open("rb") as handle:
            payload = pickle.load(handle)
        overlap = set(combined).intersection(payload)
        if overlap:
            raise RuntimeError(f"Duplicate class IDs across cluster chunks: {sorted(overlap)}")
        combined.update(payload)
        cluster_hashes[cluster_path.name] = sha256(cluster_path)
        provenance_hashes[str(provenance_path.resolve())] = sha256(provenance_path)
        rows.extend(load_jsonl(provenance_path))

    if sorted(combined) != list(range(args.classes)):
        raise RuntimeError("Cluster chunks do not cover every class ID exactly once")
    for class_id, value in combined.items():
        array = np.asarray(value)
        if array.shape != (args.ipc, 4 * 128 * 128):
            raise RuntimeError(
                f"Class {class_id} has shape {array.shape}; expected VAE guidance {(args.ipc, 65536)}"
            )
        if not np.isfinite(array).all():
            raise RuntimeError(f"Class {class_id} contains NaN or Inf")

    class_dirs = sorted(path for path in (args.data_dir / "train").iterdir() if path.is_dir())
    if len(class_dirs) != args.classes:
        raise RuntimeError(f"Source class count {len(class_dirs)} != {args.classes}")
    expected_paths = {
        str(path.resolve())
        for class_dir in class_dirs
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    actual_paths = [str(Path(row["source_path"]).resolve()) for row in rows]
    if len(actual_paths) != len(set(actual_paths)):
        raise RuntimeError("Per-image provenance contains duplicate source paths")
    if set(actual_paths) != expected_paths:
        raise RuntimeError("Per-image provenance does not exactly cover the source ImageFolder")

    by_class = defaultdict(list)
    for row in rows:
        if row.get("schema_version") != 1:
            raise RuntimeError("Unsupported per-image provenance schema")
        if row.get("feature_space") != args.feature_space:
            raise RuntimeError(f"Wrong feature-space provenance: {row['source_path']}")
        class_id = int(row["class_id"])
        if not 0 <= class_id < args.classes:
            raise RuntimeError(f"Invalid class ID {class_id}")
        if row["class_folder"] != class_dirs[class_id].name:
            raise RuntimeError(f"Class-folder mismatch for {row['source_path']}")
        probability = row["initial_hdbscan_membership_probability"]
        if probability is not None and not 0.0 <= float(probability) <= 1.0:
            raise RuntimeError(f"Invalid membership probability for {row['source_path']}")
        origin = row["final_cluster_origin"]
        if origin is not None and origin not in FINAL_ORIGINS:
            raise RuntimeError(f"Unknown final cluster origin {origin!r}")
        by_class[class_id].append(row)

    class_summary = {}
    origin_counts = Counter()
    disposition_counts = Counter()
    total_noise = 0
    total_selected_noise = 0
    for class_id, class_dir in enumerate(class_dirs):
        class_rows = by_class[class_id]
        if not class_rows:
            raise RuntimeError(f"No provenance rows for class {class_id}")
        initial_cluster_counts = {int(row["initial_hdbscan_cluster_count"]) for row in class_rows}
        if len(initial_cluster_counts) != 1:
            raise RuntimeError(f"Inconsistent initial cluster count for class {class_id}")
        selected = [row for row in class_rows if row["selected_as_representative"]]
        slots = sorted(int(row["representative_slot"]) for row in selected)
        if slots != list(range(args.ipc)):
            raise RuntimeError(f"Class {class_id} representative slots {slots} != IPC={args.ipc}")
        if len({row["final_cluster_id"] for row in selected}) != args.ipc:
            raise RuntimeError(f"Class {class_id} representatives do not map to unique final clusters")
        noise = sum(bool(row["initial_hdbscan_is_noise"]) for row in class_rows)
        selected_noise = sum(bool(row["initial_hdbscan_is_noise"]) for row in selected)
        origins = Counter(row["representative_origin"] for row in selected)
        dispositions = Counter(row["final_disposition"] for row in class_rows)
        origin_counts.update(origins)
        disposition_counts.update(dispositions)
        total_noise += noise
        total_selected_noise += selected_noise
        class_summary[str(class_id)] = {
            "class_folder": class_dir.name,
            "source_images": len(class_rows),
            "initial_hdbscan_clusters": initial_cluster_counts.pop(),
            "initial_noise_images": noise,
            "initial_noise_rate": noise / len(class_rows),
            "retained_final_cluster_images": sum(
                row["final_cluster_id"] is not None for row in class_rows
            ),
            "excluded_images": sum(row["final_cluster_id"] is None for row in class_rows),
            "representatives": len(selected),
            "selected_initial_noise": selected_noise,
            "representative_origins": dict(sorted(origins.items())),
            "final_dispositions": dict(sorted(dispositions.items())),
        }

    membership = [
        float(row["initial_hdbscan_membership_probability"])
        for row in rows
        if row["initial_hdbscan_membership_probability"] is not None
    ]
    outlier_scores = [
        float(row["initial_hdbscan_outlier_score"])
        for row in rows
        if row["initial_hdbscan_outlier_score"] is not None
    ]
    noise_ranking = sorted(
        (
            {
                "class_id": int(class_id),
                "class_folder": summary["class_folder"],
                "source_images": summary["source_images"],
                "initial_noise_images": summary["initial_noise_images"],
                "initial_noise_rate": summary["initial_noise_rate"],
            }
            for class_id, summary in class_summary.items()
        ),
        key=lambda item: (-item["initial_noise_rate"], -item["initial_noise_images"], item["class_id"]),
    )
    result = {
        "status": "complete",
        "classes": args.classes,
        "ipc": args.ipc,
        "feature_space": args.feature_space,
        "guidance_feature": "path-selected SDXL VAE latent",
        "guidance_shape_per_class": [args.ipc, 65536],
        "cluster_chunks": chunks,
        "cluster_chunk_sha256": cluster_hashes,
        "per_image_provenance_sha256": provenance_hashes,
        "per_image_provenance_rows": len(rows),
        "overall": {
            "source_images": len(rows),
            "initial_noise_images": total_noise,
            "initial_noise_rate": total_noise / len(rows),
            "classes_with_noise": sum(
                summary["initial_noise_images"] > 0 for summary in class_summary.values()
            ),
            "classes_all_noise": sum(
                summary["initial_noise_images"] == summary["source_images"]
                for summary in class_summary.values()
            ),
            "selected_representatives": args.classes * args.ipc,
            "selected_initial_noise": total_selected_noise,
            "representative_origins": dict(sorted(origin_counts.items())),
            "final_dispositions": dict(sorted(disposition_counts.items())),
            "membership_probability_mean": float(np.mean(membership)) if membership else None,
            "outlier_score_mean": float(np.mean(outlier_scores)) if outlier_scores else None,
        },
        "highest_initial_noise_rate_classes": noise_ranking[:20],
        "by_class": class_summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
