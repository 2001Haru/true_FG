"""Verify CoDA cluster outputs are VAE guidance latents for every class."""

import argparse
import hashlib
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-dir", required=True, type=Path)
    parser.add_argument("--classes", required=True, type=int)
    parser.add_argument("--ipc", required=True, type=int)
    parser.add_argument("--feature-space", required=True, choices=("vae", "dinov2"))
    parser.add_argument("--n-neighbors", required=True, type=int)
    parser.add_argument("--min-cluster-size", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    chunks = math.ceil(args.classes / 10)
    combined = {}
    hashes = {}
    for chunk_id in range(chunks):
        path = args.cluster_dir / (
            f"{args.ipc}_n_{args.n_neighbors}_s_{args.min_cluster_size}_saved_clusters_{chunk_id}.pkl"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        overlap = set(combined).intersection(payload)
        if overlap:
            raise RuntimeError(f"Duplicate class IDs across cluster chunks: {sorted(overlap)}")
        combined.update(payload)
        hashes[path.name] = sha256(path)
    if sorted(combined) != list(range(args.classes)):
        raise RuntimeError("Cluster chunks do not cover every class ID exactly once")
    shapes = set()
    for class_id, value in combined.items():
        array = np.asarray(value)
        shapes.add(tuple(array.shape))
        if array.shape != (args.ipc, 4 * 128 * 128):
            raise RuntimeError(
                f"Class {class_id} has shape {array.shape}; expected VAE guidance {(args.ipc, 65536)}"
            )
        if not np.isfinite(array).all():
            raise RuntimeError(f"Class {class_id} contains NaN or Inf")
    result = {
        "status": "complete",
        "classes": args.classes,
        "ipc": args.ipc,
        "feature_space": args.feature_space,
        "guidance_feature": "path-selected SDXL VAE latent",
        "guidance_shape_per_class": [args.ipc, 65536],
        "cluster_chunks": chunks,
        "chunk_sha256": hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
