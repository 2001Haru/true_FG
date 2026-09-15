"""Measure classwise facility-location coverage of Aircraft IPC3 selections."""

import argparse
import json
import math
import pickle
import statistics
from pathlib import Path

import numpy as np


def load_cache(root, classes=100):
    cached = {}
    for chunk_id in range(math.ceil(classes / 10)):
        with (root / f"original_features_cache.pkl_{chunk_id}").open("rb") as handle:
            payload = pickle.load(handle)
        for class_id, features in payload["features"].items():
            cached[int(class_id)] = (payload["paths"][class_id], features)
    if sorted(cached) != list(range(classes)):
        raise RuntimeError("feature cache does not cover all classes")
    result = {}
    for class_id, (paths, features) in cached.items():
        matrix = np.stack([np.asarray(feature, dtype=np.float64) for feature in features])
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        result[class_id] = ([str(Path(path).resolve()) for path in paths], matrix)
    return result


def evaluate(cache, manifest_path):
    manifest = json.loads(manifest_path.read_text())
    selected = {}
    for row in manifest["images"]:
        selected.setdefault(int(row["class_id"]), []).append(str(Path(row["source_path"]).resolve()))
    class_scores, all_maxima = [], []
    for class_id in range(100):
        paths, matrix = cache[class_id]
        index = {path: offset for offset, path in enumerate(paths)}
        chosen_paths = selected[class_id]
        if len(chosen_paths) != 3 or len(set(chosen_paths)) != 3:
            raise RuntimeError(f"class {class_id} does not have three unique selections")
        try:
            chosen = matrix[[index[path] for path in chosen_paths]]
        except KeyError as error:
            raise RuntimeError(f"selection absent from cache: {error}") from error
        maxima = (matrix @ chosen.T).max(axis=1)
        class_scores.append(float(maxima.mean()))
        all_maxima.extend(maxima.tolist())
    return {
        "macro_class_mean_max_cosine": statistics.mean(class_scores),
        "macro_class_sample_sd": statistics.stdev(class_scores),
        "macro_class_min": min(class_scores),
        "macro_class_max": max(class_scores),
        "micro_image_mean_max_cosine": statistics.mean(all_maxima),
        "macro_mean_nearest_selected_cosine_distance": 1 - statistics.mean(class_scores),
        "class_scores": class_scores,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--global-manifest", required=True, type=Path)
    p.add_argument("--kmeans-manifests", required=True, nargs=3, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    cache = load_cache(a.cache_dir)
    groups = {"global_center_top3": evaluate(cache, a.global_manifest)}
    for seed, path in enumerate(a.kmeans_manifests):
        groups[f"spherical_kmeans3_rseed{seed}"] = evaluate(cache, path)
    kmeans_scores = [groups[f"spherical_kmeans3_rseed{seed}"]["macro_class_mean_max_cosine"] for seed in range(3)]
    result = {
        "status": "complete",
        "metric": "for each class, mean over all real train images of max cosine similarity to the three selected images; then equal-weight mean over 100 classes",
        "feature_geometry": "frozen DINOv2-base CLS, L2-normalized, deterministic Resize256-CenterCrop224",
        "candidate_universe": "all 6667 Aircraft training images, within true class",
        "groups": groups,
        "kmeans_across_selection_seeds": {
            "mean_macro_coverage": statistics.mean(kmeans_scores),
            "sample_sd_across_selection_seeds": statistics.stdev(kmeans_scores),
            "values": kmeans_scores,
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
