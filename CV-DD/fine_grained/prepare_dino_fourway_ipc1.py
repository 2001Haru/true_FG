"""Select IPC1 real images using five controlled DINO cosine geometries."""

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
from pathlib import Path

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DETERMINISTIC_METHODS = (
    "centroid",
    "rival_facing_edge",
    "outward_edge",
    "edge_high_margin",
)
RANDOM_SEEDS = (0, 1, 2)
SHELL_LOW = 0.70
SHELL_HIGH = 0.95


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_random_key(seed: int, relative_path: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{relative_path}".encode("utf-8")).digest()


def stable_argmin(indices, value, paths):
    return min(indices, key=lambda index: (float(value[index]), paths[index]))


def stable_argmax(indices, value, paths):
    return min(indices, key=lambda index: (-float(value[index]), paths[index]))


def materialize(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if mode == "symlink":
            if not destination.is_symlink() or destination.resolve() != source.resolve():
                raise RuntimeError(f"selection destination collision: {destination}")
        elif destination.is_symlink() or sha256(destination) != sha256(source):
            raise RuntimeError(f"selection destination collision: {destination}")
        return
    if mode == "symlink":
        os.symlink(source.resolve(), destination)
    else:
        shutil.copy2(source, destination)


def load_cache(cache_dir: Path, classes: int):
    cached = {}
    chunk_hashes = {}
    for chunk_id in range(math.ceil(classes / 10)):
        path = cache_dir / f"original_features_cache.pkl_{chunk_id}"
        if not path.is_file():
            raise FileNotFoundError(path)
        chunk_hashes[path.name] = sha256(path)
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        for class_id, features in payload["features"].items():
            class_id = int(class_id)
            if class_id in cached:
                raise RuntimeError(f"duplicate cached class {class_id}")
            cached[class_id] = (payload["paths"][class_id], features)
    if sorted(cached) != list(range(classes)):
        raise RuntimeError("DINO cache does not cover every class exactly once")
    return cached, chunk_hashes


def selected_tree_sha256(records: list[dict], output_root: Path) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda row: row["selected_path"]):
        relative = Path(record["selected_path"]).relative_to(output_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(bytes.fromhex(record["source_sha256"]))
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--feature-audit", required=True, type=Path)
    parser.add_argument("--generation-config", required=True, type=Path)
    parser.add_argument("--dino-model-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--classes", required=True, type=int)
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    args = parser.parse_args()

    source_root = (args.data_dir / "train").resolve()
    class_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
    if len(class_dirs) != args.classes:
        raise RuntimeError(f"source class count {len(class_dirs)} != {args.classes}")
    feature_audit = json.loads(args.feature_audit.read_text(encoding="utf-8"))
    expected_audit = {
        "status": "complete",
        "feature_space": "dinov2",
        "classes": args.classes,
        "feature_dimension": 768,
        "cache_dir": str(args.cache_dir.resolve()),
    }
    for key, value in expected_audit.items():
        if feature_audit.get(key) != value:
            raise RuntimeError(
                f"feature audit {key}={feature_audit.get(key)!r}, expected {value!r}"
            )
    generation_config = json.loads(args.generation_config.read_text(encoding="utf-8"))
    model_path = args.dino_model_root / "model.safetensors"
    config_path = args.dino_model_root / "config.json"
    preprocessor_path = args.dino_model_root / "preprocessor_config.json"
    for path in (model_path, config_path, preprocessor_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_generation = {
        "status": "frozen",
        "dataset": args.dataset_name,
        "classes": args.classes,
        "feature_space": "dinov2",
        "data_dir": str(args.data_dir.resolve()),
        "dino_model_root": str(args.dino_model_root.resolve()),
        "dino_model_sha256": sha256(model_path),
        "dino_config_sha256": sha256(config_path),
        "dino_preprocessor_sha256": sha256(preprocessor_path),
        "clustering_feature_encoder": (
            "DINOv2-base final normalized CLS token (768D), Resize256+CenterCrop224"
        ),
    }
    for key, value in expected_generation.items():
        if generation_config.get(key) != value:
            raise RuntimeError(
                f"generation config {key}={generation_config.get(key)!r}, expected {value!r}"
            )
    extraction_source = args.repo_root / "CoDA" / "get_features.py"
    recorded_source_hash = generation_config.get("source_sha256", {}).get("CoDA/get_features.py")
    if recorded_source_hash != sha256(extraction_source):
        raise RuntimeError("current DINO extraction source differs from cache provenance")

    cached, chunk_hashes = load_cache(args.cache_dir, args.classes)
    all_z = []
    all_paths = []
    all_class_ids = []
    per_class_indices = {}
    source_paths_expected = set()
    for class_id, class_dir in enumerate(class_dirs):
        expected_paths = sorted(
            str(path.resolve())
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        source_paths_expected.update(expected_paths)
        paths, features = cached[class_id]
        normalized_paths = [str(Path(path).resolve()) for path in paths]
        if len(normalized_paths) != len(set(normalized_paths)) or sorted(normalized_paths) != expected_paths:
            raise RuntimeError(f"cache/source path mismatch for class {class_id}")
        matrix = np.stack([np.asarray(feature, dtype=np.float64) for feature in features])
        if matrix.shape != (len(paths), 768) or not np.isfinite(matrix).all():
            raise RuntimeError(f"invalid DINO feature matrix for class {class_id}: {matrix.shape}")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms <= 0):
            raise RuntimeError(f"zero-norm DINO feature in class {class_id}")
        z = matrix / norms
        start = len(all_paths)
        indices = list(range(start, start + len(paths)))
        per_class_indices[class_id] = indices
        all_z.extend(z)
        all_paths.extend(normalized_paths)
        all_class_ids.extend([class_id] * len(paths))
    if len(all_paths) != feature_audit.get("images") or set(all_paths) != source_paths_expected:
        raise RuntimeError("combined cache does not exactly cover the train split")
    z_matrix = np.asarray(all_z, dtype=np.float64)
    prototypes = []
    for class_id in range(args.classes):
        summed = z_matrix[per_class_indices[class_id]].sum(axis=0)
        norm = np.linalg.norm(summed)
        if norm <= 0:
            raise RuntimeError(f"zero-norm prototype for class {class_id}")
        prototypes.append(summed / norm)
    prototypes = np.asarray(prototypes, dtype=np.float64)

    own_similarity = np.empty(len(all_paths), dtype=np.float64)
    rival_similarity = np.empty(len(all_paths), dtype=np.float64)
    rival_class = np.empty(len(all_paths), dtype=np.int64)
    radial_distance = np.empty(len(all_paths), dtype=np.float64)
    margin = np.empty(len(all_paths), dtype=np.float64)
    percentile = np.empty(len(all_paths), dtype=np.float64)
    shell = np.zeros(len(all_paths), dtype=bool)
    selections = {method: {} for method in DETERMINISTIC_METHODS}
    random_selections = {seed: {} for seed in RANDOM_SEEDS}
    class_summaries = []
    for class_id in range(args.classes):
        indices = per_class_indices[class_id]
        similarities = z_matrix[indices] @ prototypes.T
        local_own = similarities[:, class_id].copy()
        similarities[:, class_id] = -np.inf
        local_rival_class = similarities.argmax(axis=1)
        local_rival = similarities[np.arange(len(indices)), local_rival_class]
        local_radial = 1.0 - local_own
        local_margin = local_own - local_rival
        for local_index, global_index in enumerate(indices):
            own_similarity[global_index] = local_own[local_index]
            rival_similarity[global_index] = local_rival[local_index]
            rival_class[global_index] = local_rival_class[local_index]
            radial_distance[global_index] = local_radial[local_index]
            margin[global_index] = local_margin[local_index]
        ranked = sorted(indices, key=lambda index: (float(radial_distance[index]), all_paths[index]))
        denominator = max(len(ranked) - 1, 1)
        for rank, global_index in enumerate(ranked):
            percentile[global_index] = rank / denominator
            shell[global_index] = SHELL_LOW <= percentile[global_index] <= SHELL_HIGH
        shell_indices = [index for index in indices if shell[index]]
        if not shell_indices:
            raise RuntimeError(f"empty pure-radial edge shell for class {class_id}")
        selections["centroid"][class_id] = stable_argmax(indices, own_similarity, all_paths)
        selections["rival_facing_edge"][class_id] = stable_argmax(
            shell_indices, rival_similarity, all_paths
        )
        selections["outward_edge"][class_id] = stable_argmin(
            shell_indices, rival_similarity, all_paths
        )
        selections["edge_high_margin"][class_id] = stable_argmax(
            shell_indices, margin, all_paths
        )
        for seed in RANDOM_SEEDS:
            random_selections[seed][class_id] = min(
                indices,
                key=lambda index: stable_random_key(
                    seed, Path(all_paths[index]).relative_to(source_root).as_posix()
                ),
            )
        outward = selections["outward_edge"][class_id]
        high_margin = selections["edge_high_margin"][class_id]
        class_summaries.append(
            {
                "class_id": class_id,
                "class_folder": class_dirs[class_id].name,
                "images": len(indices),
                "nonnegative_margin_images": int(
                    sum(bool(margin[index] >= 0) for index in indices)
                ),
                "shell_images": len(shell_indices),
                "outward_matches_edge_high_margin": outward == high_margin,
                "selected_indices": {
                    method: selections[method][class_id]
                    for method in DETERMINISTIC_METHODS
                },
            }
        )

    image_records = [
        {
            "global_index": index,
            "source_path": all_paths[index],
            "class_id": int(all_class_ids[index]),
            "class_folder": class_dirs[all_class_ids[index]].name,
            "own_centroid_similarity": float(own_similarity[index]),
            "radial_cosine_distance": float(radial_distance[index]),
            "nearest_rival_similarity": float(rival_similarity[index]),
            "nearest_rival_class_id": int(rival_class[index]),
            "nearest_rival_class_folder": class_dirs[int(rival_class[index])].name,
            "prototype_margin": float(margin[index]),
            "within_class_radial_percentile": float(percentile[index]),
            "in_edge_shell": bool(shell[index]),
            "selected_by": [
                method
                for method in DETERMINISTIC_METHODS
                if selections[method].get(all_class_ids[index]) == index
            ]
            + [
                f"random_rseed{seed}"
                for seed in RANDOM_SEEDS
                if random_selections[seed].get(all_class_ids[index]) == index
            ],
        }
        for index in range(len(all_paths))
    ]
    audit = {
        "status": "complete",
        "dataset": args.dataset_name,
        "classes": args.classes,
        "ipc": 1,
        "embedding": "DINOv2-base pooler_output (final LayerNorm CLS), L2-normalized",
        "geometry": "cosine similarity on unit-normalized embeddings and normalized class means",
        "dino_transform": "Resize shorter edge 256 (bicubic)->CenterCrop224->ImageNet normalization",
        "random_augmentation_in_selection": False,
        "dino_model_root": str(args.dino_model_root.resolve()),
        "dino_model_sha256": sha256(model_path),
        "dino_config_sha256": sha256(config_path),
        "dino_preprocessor_sha256": sha256(preprocessor_path),
        "feature_audit": str(args.feature_audit.resolve()),
        "feature_audit_sha256": sha256(args.feature_audit),
        "generation_config": str(args.generation_config.resolve()),
        "generation_config_sha256": sha256(args.generation_config),
        "feature_extraction_source": str(extraction_source.resolve()),
        "feature_extraction_source_sha256": sha256(extraction_source),
        "cache_chunk_sha256": chunk_hashes,
        "source_root": str(source_root),
        "source_images": len(all_paths),
        "embedding_dimension": 768,
        "embedding_dtype_for_geometry": "float64",
        "prototype_sha256": hashlib.sha256(
            prototypes.astype("<f8", copy=False).tobytes()
        ).hexdigest(),
        "radial_percentile_definition": "zero-based stable rank/(class_size-1), ascending r; path tie-break",
        "shell": {
            "prototype_correctness_filter": False,
            "radial_percentile_low_inclusive": SHELL_LOW,
            "radial_percentile_high_inclusive": SHELL_HIGH,
            "definition": "0.70 <= within-class percentile(r) <= 0.95",
            "revision_reason": (
                "Removed m>=0 before post-eval because it measures correctness of an "
                "unadapted DINO nearest-centroid proxy, not validity of the ground-truth label"
            ),
        },
        "directional_definitions": {
            "rival_facing_edge": "maximum nearest-rival similarity b within shell",
            "outward_edge": "minimum nearest-rival similarity b within shell",
            "edge_high_margin": "maximum prototype margin m=a-b within shell",
        },
        "outward_edge_high_margin_overlap_classes": sum(
            row["outward_matches_edge_high_margin"] for row in class_summaries
        ),
        "outward_edge_high_margin_overlap_rate": (
            sum(row["outward_matches_edge_high_margin"] for row in class_summaries)
            / args.classes
        ),
        "class_summaries": class_summaries,
        "images": image_records,
    }
    write_json(args.audit_output, audit)
    arms = {method: selections[method] for method in DETERMINISTIC_METHODS}
    arms.update({f"random_rseed{seed}": random_selections[seed] for seed in RANDOM_SEEDS})
    manifest_paths = {}
    for arm, by_class in arms.items():
        selection_root = args.output_root / arm
        selected_records = []
        for class_id in range(args.classes):
            index = by_class[class_id]
            source = Path(all_paths[index])
            destination = selection_root / class_dirs[class_id].name / source.name
            materialize(source, destination, args.link_mode)
            selected_records.append(
                {
                    "class_id": class_id,
                    "class_folder": class_dirs[class_id].name,
                    "source_path": str(source),
                    "selected_path": str(destination.absolute()),
                    "source_sha256": sha256(source),
                    "own_centroid_similarity": float(own_similarity[index]),
                    "radial_cosine_distance": float(radial_distance[index]),
                    "nearest_rival_similarity": float(rival_similarity[index]),
                    "nearest_rival_class_id": int(rival_class[index]),
                    "nearest_rival_class_folder": class_dirs[int(rival_class[index])].name,
                    "prototype_margin": float(margin[index]),
                    "within_class_radial_percentile": float(percentile[index]),
                    "in_edge_shell": bool(shell[index]),
                }
            )
        actual = {
            path.relative_to(selection_root).as_posix()
            for class_dir in selection_root.iterdir()
            if class_dir.is_dir()
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        expected = {
            Path(row["selected_path"]).relative_to(selection_root).as_posix()
            for row in selected_records
        }
        if actual != expected or len(actual) != args.classes:
            raise RuntimeError(f"selected ImageFolder audit failed for {arm}")
        random_seed = int(arm.removeprefix("random_rseed")) if arm.startswith("random_rseed") else None
        manifest = {
            "status": "complete",
            "experiment": "dino_fivearm_ipc1",
            "dataset": args.dataset_name,
            "classes": args.classes,
            "ipc": 1,
            "selection_method": "random" if random_seed is not None else arm,
            "selection_arm": arm,
            "selection_seed": random_seed,
            "selection_root": str(selection_root.resolve()),
            "selection_images": args.classes,
            "selection_audit": str(args.audit_output.resolve()),
            "selection_audit_sha256": sha256(args.audit_output),
            "selected_tree_sha256": selected_tree_sha256(selected_records, selection_root),
            "materialization": args.link_mode,
            "images": selected_records,
        }
        manifest_path = args.output_root / "manifests" / f"{arm}.json"
        write_json(manifest_path, manifest)
        manifest_paths[arm] = str(manifest_path.resolve())
    print(
        json.dumps(
            {
                "status": "complete",
                "dataset": args.dataset_name,
                "source_images": len(all_paths),
                "arms": manifest_paths,
                "outward_edge_high_margin_overlap_rate": audit[
                    "outward_edge_high_margin_overlap_rate"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
