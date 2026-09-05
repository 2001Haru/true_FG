"""Build audited IPC2 real-image arms from the frozen IPC1 DINO geometry."""

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from prepare_dino_fourway_ipc1 import load_cache


SELECTION_SEEDS = (0, 1, 2)
STOCHASTIC_METHODS = (
    "random_ipc2",
    "spherical_kmeans2",
    "center_plus_random",
    "center_plus_shell_random",
)
DETERMINISTIC_METHODS = (
    "center_plus_outward",
    "center_plus_high_margin",
    "center_plus_rival_facing",
    "global_center_top2",
)
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 100


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def random_key(seed: int, relative_path: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{relative_path}".encode("utf-8")).digest()


def class_rng(seed: int, dataset: str, class_id: int, init_id: int) -> np.random.Generator:
    digest = hashlib.sha256(
        f"spherical-kmeans2-v1\0{seed}\0{dataset}\0{class_id}\0{init_id}".encode()
    ).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def spherical_kmeans_once(z: np.ndarray, rng: np.random.Generator):
    count = len(z)
    first = int(rng.integers(count))
    distance = np.clip(1.0 - z @ z[first], 0.0, None)
    if distance.sum() <= 0:
        second = (first + 1) % count
    else:
        second = int(rng.choice(count, p=distance / distance.sum()))
    centers = np.stack((z[first], z[second])).copy()
    assignments = None
    empty_cluster_events = 0
    for iteration in range(1, KMEANS_MAX_ITER + 1):
        similarities = z @ centers.T
        new_assignments = similarities.argmax(axis=1)
        if len(set(map(int, new_assignments))) != 2:
            empty_cluster_events += 1
            weakest = int(np.argmin(similarities.max(axis=1)))
            empty = 1 - int(new_assignments[weakest])
            new_assignments[weakest] = empty
        new_centers = []
        for cluster_id in range(2):
            summed = z[new_assignments == cluster_id].sum(axis=0)
            norm = np.linalg.norm(summed)
            if norm <= 0:
                raise RuntimeError("zero-norm spherical K-means center")
            new_centers.append(summed / norm)
        new_centers = np.asarray(new_centers)
        converged = assignments is not None and np.array_equal(new_assignments, assignments)
        assignments = new_assignments
        centers = new_centers
        if converged:
            break
    best_similarity = (z @ centers.T).max(axis=1)
    inertia = float(np.sum(1.0 - best_similarity))
    return assignments, centers, inertia, iteration, empty_cluster_events


def spherical_kmeans(
    z: np.ndarray, paths: list[str], seed: int, dataset: str, class_id: int
):
    solutions = []
    for init_id in range(KMEANS_N_INIT):
        assignments, centers, inertia, iterations, empty_events = spherical_kmeans_once(
            z, class_rng(seed, dataset, class_id, init_id)
        )
        representatives = []
        for cluster_id in range(2):
            indices = np.flatnonzero(assignments == cluster_id)
            representative = min(
                map(int, indices),
                key=lambda index: (-float(z[index] @ centers[cluster_id]), paths[index]),
            )
            representatives.append(representative)
        canonical_paths = tuple(sorted(paths[index] for index in representatives))
        solutions.append(
            (
                inertia,
                canonical_paths,
                init_id,
                assignments,
                centers,
                representatives,
                iterations,
                empty_events,
            )
        )
    best = min(solutions, key=lambda item: (item[0], item[1], item[2]))
    inertia, _, init_id, assignments, centers, representatives, iterations, empty_events = best
    records = []
    for cluster_id, representative in enumerate(representatives):
        records.append(
            {
                "local_index": representative,
                "cluster_id_before_canonicalization": cluster_id,
                "cluster_size": int((assignments == cluster_id).sum()),
                "representative_cluster_similarity": float(
                    z[representative] @ centers[cluster_id]
                ),
            }
        )
    records.sort(key=lambda row: paths[row["local_index"]])
    for canonical_id, row in enumerate(records):
        row["cluster_id"] = canonical_id
    return {
        "representatives": records,
        "selected_init": init_id,
        "cosine_inertia": inertia,
        "iterations": iterations,
        "empty_cluster_events": empty_events,
    }


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


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def tree_sha256(records: list[dict], root: Path) -> str:
    digest = hashlib.sha256()
    for row in sorted(records, key=lambda item: item["selected_path"]):
        digest.update(Path(row["selected_path"]).relative_to(root).as_posix().encode())
        digest.update(bytes.fromhex(row["source_sha256"]))
    return digest.hexdigest()


def load_manifest(path: Path, classes: int) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or payload.get("selection_images") != classes:
        raise RuntimeError(f"invalid parent IPC1 manifest: {path}")
    rows = {int(row["class_id"]): row for row in payload["images"]}
    if sorted(rows) != list(range(classes)):
        raise RuntimeError(f"parent manifest does not cover every class: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-experiment-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--classes", required=True, type=int)
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    args = parser.parse_args()

    parent_selection = args.parent_experiment_root / "selections" / args.dataset_name
    geometry_path = (
        args.parent_experiment_root / "selection_audits" / f"{args.dataset_name}.json"
    )
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if (
        geometry.get("status") != "complete"
        or geometry.get("classes") != args.classes
        or geometry.get("ipc") != 1
        or geometry.get("shell", {}).get("prototype_correctness_filter") is not False
    ):
        raise RuntimeError("parent pure-radial IPC1 geometry audit is invalid")
    image_rows = {row["source_path"]: row for row in geometry["images"]}
    by_class_geometry = defaultdict(list)
    for row in geometry["images"]:
        by_class_geometry[int(row["class_id"])].append(row)
    center = load_manifest(parent_selection / "manifests" / "centroid.json", args.classes)
    outward = load_manifest(parent_selection / "manifests" / "outward_edge.json", args.classes)
    high_margin = load_manifest(
        parent_selection / "manifests" / "edge_high_margin.json", args.classes
    )
    rival = load_manifest(
        parent_selection / "manifests" / "rival_facing_edge.json", args.classes
    )
    global_random = {
        seed: load_manifest(
            parent_selection / "manifests" / f"random_rseed{seed}.json", args.classes
        )
        for seed in SELECTION_SEEDS
    }
    shell_random = {
        seed: load_manifest(
            parent_selection / "manifests" / f"shell_random_rseed{seed}.json",
            args.classes,
        )
        for seed in SELECTION_SEEDS
    }

    cache_dir = Path(json.loads(Path(geometry["feature_audit"]).read_text())["cache_dir"])
    cached, cache_hashes = load_cache(cache_dir, args.classes)
    if cache_hashes != geometry["cache_chunk_sha256"]:
        raise RuntimeError("DINO cache differs from frozen IPC1 geometry provenance")
    source_root = Path(geometry["source_root"])
    per_class_paths = {}
    per_class_z = {}
    for class_id in range(args.classes):
        paths, features = cached[class_id]
        normalized_paths = [str(Path(path).resolve()) for path in paths]
        matrix = np.stack([np.asarray(feature, dtype=np.float64) for feature in features])
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        if set(normalized_paths) != {
            row["source_path"] for row in by_class_geometry[class_id]
        }:
            raise RuntimeError(f"DINO cache path mismatch in class {class_id}")
        per_class_paths[class_id] = normalized_paths
        per_class_z[class_id] = matrix

    arms = {}
    kmeans_audit = {seed: {} for seed in SELECTION_SEEDS}
    for seed in SELECTION_SEEDS:
        random_rows = {}
        center_random_rows = {}
        center_shell_rows = {}
        kmeans_rows = {}
        for class_id in range(args.classes):
            paths = per_class_paths[class_id]
            ranked = sorted(
                paths,
                key=lambda path: random_key(
                    seed, Path(path).relative_to(source_root).as_posix()
                ),
            )
            random_rows[class_id] = [
                (ranked[0], "random_rank_1", {}),
                (ranked[1], "random_rank_2", {}),
            ]
            center_path = center[class_id]["source_path"]
            remaining = [path for path in ranked if path != center_path]
            center_random_rows[class_id] = [
                (center_path, "fixed_ipc1_center", {}),
                (remaining[0], "random_increment", {}),
            ]
            center_shell_rows[class_id] = [
                (center_path, "fixed_ipc1_center", {}),
                (
                    shell_random[seed][class_id]["source_path"],
                    "frozen_ipc1_shell_random_increment",
                    {},
                ),
            ]
            solution = spherical_kmeans(
                per_class_z[class_id], paths, seed, args.dataset_name, class_id
            )
            kmeans_audit[seed][class_id] = {
                key: value for key, value in solution.items() if key != "representatives"
            }
            kmeans_rows[class_id] = [
                (
                    paths[row["local_index"]],
                    f"spherical_cluster_{row['cluster_id']}_representative",
                    {key: value for key, value in row.items() if key != "local_index"},
                )
                for row in solution["representatives"]
            ]
        arms[f"random_ipc2_rseed{seed}"] = random_rows
        arms[f"spherical_kmeans2_rseed{seed}"] = kmeans_rows
        arms[f"center_plus_random_rseed{seed}"] = center_random_rows
        arms[f"center_plus_shell_random_rseed{seed}"] = center_shell_rows

    deterministic_sources = {
        "center_plus_outward": outward,
        "center_plus_high_margin": high_margin,
        "center_plus_rival_facing": rival,
    }
    for arm, increment in deterministic_sources.items():
        arms[arm] = {
            class_id: [
                (center[class_id]["source_path"], "fixed_ipc1_center", {}),
                (
                    increment[class_id]["source_path"],
                    f"frozen_ipc1_{arm.removeprefix('center_plus_')}_increment",
                    {},
                ),
            ]
            for class_id in range(args.classes)
        }
    arms["global_center_top2"] = {}
    for class_id in range(args.classes):
        ranked = sorted(
            by_class_geometry[class_id],
            key=lambda row: (-row["own_centroid_similarity"], row["source_path"]),
        )
        if ranked[0]["source_path"] != center[class_id]["source_path"]:
            raise RuntimeError(f"IPC1 center mismatch in class {class_id}")
        arms["global_center_top2"][class_id] = [
            (ranked[0]["source_path"], "global_center_rank_1_fixed_ipc1_center", {}),
            (ranked[1]["source_path"], "global_center_rank_2", {}),
        ]

    center_collisions = []
    duplicate_pairs = []
    for arm, classes in arms.items():
        for class_id, pair in classes.items():
            if pair[0][0] == pair[1][0]:
                duplicate_pairs.append({"arm": arm, "class_id": class_id, "path": pair[0][0]})
            if arm.startswith("center_plus_"):
                expected_center = center[class_id]["source_path"]
                if pair[0][0] != expected_center or pair[1][0] == expected_center:
                    center_collisions.append(
                        {"arm": arm, "class_id": class_id, "pair": [pair[0][0], pair[1][0]]}
                    )
    if center_collisions or duplicate_pairs:
        atomic_json(
            args.audit_output,
            {
                "status": "invalid",
                "center_collisions": center_collisions,
                "duplicate_pairs": duplicate_pairs,
            },
        )
        raise RuntimeError("IPC2 selection contains a center collision or duplicate pair")

    manifests = {}
    arm_summaries = {}
    for arm, classes in arms.items():
        selection_root = args.output_root / "selections" / args.dataset_name / arm
        records = []
        for class_id in range(args.classes):
            for slot, (source_path, role, extra) in enumerate(classes[class_id]):
                source = Path(source_path)
                geometry_row = image_rows[str(source.resolve())]
                destination = selection_root / geometry_row["class_folder"] / source.name
                materialize(source, destination, args.link_mode)
                records.append(
                    {
                        "class_id": class_id,
                        "class_folder": geometry_row["class_folder"],
                        "slot": slot,
                        "selection_role": role,
                        "source_path": str(source.resolve()),
                        "selected_path": str(destination.absolute()),
                        "source_sha256": sha256(source),
                        "own_centroid_similarity": geometry_row["own_centroid_similarity"],
                        "radial_cosine_distance": geometry_row["radial_cosine_distance"],
                        "nearest_rival_similarity": geometry_row[
                            "nearest_rival_similarity"
                        ],
                        "prototype_margin": geometry_row["prototype_margin"],
                        "within_class_radial_percentile": geometry_row[
                            "within_class_radial_percentile"
                        ],
                        "in_frozen_edge_shell": geometry_row["in_edge_shell"],
                        **extra,
                    }
                )
        actual = {
            path.relative_to(selection_root).as_posix()
            for class_dir in selection_root.iterdir()
            if class_dir.is_dir()
            for path in class_dir.iterdir()
            if path.is_file()
        }
        expected = {
            Path(row["selected_path"]).relative_to(selection_root).as_posix()
            for row in records
        }
        if actual != expected or len(actual) != 2 * args.classes:
            raise RuntimeError(f"ImageFolder audit failed for {arm}")
        method = next(
            method for method in (*STOCHASTIC_METHODS, *DETERMINISTIC_METHODS) if arm.startswith(method)
        )
        selection_seed = (
            int(arm.rsplit("rseed", 1)[1]) if "rseed" in arm else None
        )
        manifest = {
            "status": "complete",
            "experiment": "dino_ipc2_center_complement",
            "dataset": args.dataset_name,
            "classes": args.classes,
            "ipc": 2,
            "selection_method": method,
            "selection_arm": arm,
            "selection_seed": selection_seed,
            "selection_images": 2 * args.classes,
            "selection_root": str(selection_root.resolve()),
            "parent_ipc1_geometry": str(geometry_path.resolve()),
            "recomputed_geometry": False,
            "training_sample_weighting": "equal; both images mixed by the same shuffled loader",
            "selected_tree_sha256": tree_sha256(records, selection_root),
            "materialization": args.link_mode,
            "images": records,
        }
        manifest_path = (
            args.output_root / "manifests" / args.dataset_name / f"{arm}.json"
        )
        atomic_json(manifest_path, manifest)
        manifests[arm] = str(manifest_path.resolve())
        arm_summaries[arm] = {
            "classes": args.classes,
            "images": len(records),
            "duplicate_pairs": 0,
            "center_is_identical_to_ipc1_for_every_center_plus_class": (
                True if arm.startswith("center_plus_") else None
            ),
        }
    audit = {
        "status": "complete",
        "experiment": "dino_ipc2_center_complement",
        "dataset": args.dataset_name,
        "classes": args.classes,
        "ipc": 2,
        "parent_ipc1_geometry": str(geometry_path.resolve()),
        "geometry_recomputed": False,
        "center_source": "exact frozen IPC1 centroid selection",
        "edge_sources": "exact frozen IPC1 outward/high-margin/rival-facing selections",
        "shell_random_source": "exact frozen IPC1 shell-random selections",
        "spherical_kmeans": {
            "k": 2,
            "geometry": "cosine assignment on unit DINO embeddings",
            "initialization": "spherical k-means++",
            "n_init": KMEANS_N_INIT,
            "max_iterations": KMEANS_MAX_ITER,
            "center_update": "normalized mean",
            "representative": "maximum cosine similarity to its cluster center",
            "runs": kmeans_audit,
        },
        "center_collisions": [],
        "duplicate_pairs": [],
        "arm_summaries": arm_summaries,
        "manifests": manifests,
    }
    atomic_json(args.audit_output, audit)
    print(
        json.dumps(
            {
                "status": "complete",
                "dataset": args.dataset_name,
                "selection_arms": len(arms),
                "center_collisions": 0,
                "duplicate_pairs": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

