"""Validate CoDA VAE/DINO feature caches against the source ImageFolder."""

import argparse
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-space", required=True, choices=("vae", "dinov2"))
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--classes", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    class_dirs = sorted(path for path in (args.data_dir / "train").iterdir() if path.is_dir())
    if len(class_dirs) != args.classes:
        raise RuntimeError(f"Source class count {len(class_dirs)} != {args.classes}")
    chunks = math.ceil(args.classes / 10)
    cached = {}
    chunk_sizes = {}
    for chunk_id in range(chunks):
        path = args.cache_dir / f"original_features_cache.pkl_{chunk_id}"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        chunk_sizes[path.name] = path.stat().st_size
        for class_id in payload["features"]:
            if class_id in cached:
                raise RuntimeError(f"Duplicate cached class ID {class_id}")
            cached[class_id] = (
                payload["paths"][class_id],
                payload["features"][class_id],
            )
    if sorted(cached) != list(range(args.classes)):
        raise RuntimeError("Feature cache does not cover every class exactly once")
    feature_dimension = 65536 if args.feature_space == "vae" else 768
    image_count = 0
    for class_id, class_dir in enumerate(class_dirs):
        expected_paths = sorted(
            str(path.resolve()) for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        )
        paths, features = cached[class_id]
        if len(paths) != len(set(paths)):
            raise RuntimeError(f"Class {class_id} cache contains duplicate paths")
        if sorted(str(Path(path).resolve()) for path in paths) != expected_paths:
            raise RuntimeError(f"Class {class_id} cache paths differ from source ImageFolder")
        if len(features) != len(paths):
            raise RuntimeError(f"Class {class_id} feature/path count differs")
        for feature in features:
            array = np.asarray(feature)
            if array.shape != (feature_dimension,):
                raise RuntimeError(f"Class {class_id} feature shape {array.shape} != {(feature_dimension,)}")
            if not np.isfinite(array).all():
                raise RuntimeError(f"Class {class_id} contains NaN or Inf")
        image_count += len(paths)
    result = {
        "status": "complete",
        "feature_space": args.feature_space,
        "classes": args.classes,
        "images": image_count,
        "feature_dimension": feature_dimension,
        "cache_dir": str(args.cache_dir.resolve()),
        "chunk_files": chunks,
        "chunk_bytes": chunk_sizes,
        "duplicates": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
