"""Build audited Aircraft IPC3 Global-center Top3 and spherical K-means K=3 manifests."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import prepare_dino_ipc5_selection as km
from prepare_dino_fourway_ipc1 import load_cache


IPC = 3
SELECTION_SEEDS = (0, 1, 2)
EXPERIMENT = "dino_ipc3_top3_kmeans3"


def class_rng(seed: int, dataset: str, class_id: int, init_id: int):
    digest = hashlib.sha256(
        f"spherical-kmeans3-v1\0{seed}\0{dataset}\0{class_id}\0{init_id}".encode()
    ).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-experiment-root", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--dataset-name", default="A_imsize224")
    p.add_argument("--classes", default=100, type=int)
    a = p.parse_args()
    geometry_path = a.parent_experiment_root / "selection_audits" / f"{a.dataset_name}.json"
    geometry = json.loads(geometry_path.read_text())
    assert geometry["status"] == "complete" and geometry["classes"] == a.classes
    image_rows = {row["source_path"]: row for row in geometry["images"]}
    by_class = defaultdict(list)
    for row in geometry["images"]:
        by_class[int(row["class_id"])].append(row)
    cache_dir = Path(json.loads(Path(geometry["feature_audit"]).read_text())["cache_dir"])
    cached, hashes = load_cache(cache_dir, a.classes)
    assert hashes == geometry["cache_chunk_sha256"]
    paths_by_class, z_by_class = {}, {}
    for class_id in range(a.classes):
        paths, features = cached[class_id]
        paths = [str(Path(path).resolve()) for path in paths]
        z = np.stack([np.asarray(feature, dtype=np.float64) for feature in features])
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        assert len(paths) >= IPC and set(paths) == {row["source_path"] for row in by_class[class_id]}
        paths_by_class[class_id], z_by_class[class_id] = paths, z

    km.IPC = IPC
    km.class_rng = class_rng
    arms = {"global_center_top3": {}}
    runs = {seed: {} for seed in SELECTION_SEEDS}
    for class_id in range(a.classes):
        ranked = sorted(by_class[class_id], key=lambda row: (-float(row["own_centroid_similarity"]), row["source_path"]))
        arms["global_center_top3"][class_id] = [
            (row["source_path"], f"global_center_rank_{rank + 1}", {}) for rank, row in enumerate(ranked[:IPC])]
    for seed in SELECTION_SEEDS:
        selected = {}
        for class_id in range(a.classes):
            paths, z = paths_by_class[class_id], z_by_class[class_id]
            solution = km.spherical_kmeans(z, paths, seed, a.dataset_name, class_id)
            runs[seed][class_id] = {key: value for key, value in solution.items() if key != "representatives"}
            selected[class_id] = [
                (paths[row["local_index"]], f"spherical_cluster_{row['cluster_id']}_representative",
                 {key: value for key, value in row.items() if key != "local_index"})
                for row in solution["representatives"]]
        arms[f"spherical_kmeans3_rseed{seed}"] = selected

    manifests = {}
    for arm, classes in arms.items():
        records = []
        for class_id in range(a.classes):
            selected = classes[class_id]
            assert len(selected) == len({row[0] for row in selected}) == IPC
            for slot, (source_path, role, extra) in enumerate(selected):
                g = image_rows[str(Path(source_path).resolve())]
                records.append({"class_id": class_id, "class_folder": g["class_folder"], "slot": slot,
                                "selection_role": role, "source_path": str(Path(source_path).resolve()),
                                "selected_path": None, "own_centroid_similarity": g["own_centroid_similarity"],
                                "radial_cosine_distance": g["radial_cosine_distance"],
                                "nearest_rival_similarity": g["nearest_rival_similarity"],
                                "prototype_margin": g["prototype_margin"],
                                "within_class_radial_percentile": g["within_class_radial_percentile"], **extra})
        manifest = {"status": "complete", "experiment": EXPERIMENT, "dataset": a.dataset_name,
                    "classes": a.classes, "ipc": IPC,
                    "selection_method": "global_center_top3" if arm == "global_center_top3" else "spherical_kmeans3",
                    "selection_arm": arm,
                    "selection_seed": None if arm == "global_center_top3" else int(arm.rsplit("rseed", 1)[1]),
                    "selection_images": IPC * a.classes, "parent_ipc1_geometry": str(geometry_path.resolve()),
                    "recomputed_geometry": False, "training_sample_weighting": "equal",
                    "selected_path_identity_sha256": km.selection_identity(records), "materialization": "none",
                    "images": records}
        path = a.output_root / f"manifests/{a.dataset_name}/{arm}.json"
        km.atomic_json(path, manifest); manifests[arm] = str(path.resolve())
    empty = sum(run["empty_cluster_events"] for seed_runs in runs.values() for run in seed_runs.values())
    audit = {"status": "complete", "experiment": EXPERIMENT, "dataset": a.dataset_name,
             "classes": a.classes, "ipc": IPC, "parent_ipc1_geometry": str(geometry_path.resolve()),
             "geometry_recomputed": False, "spherical_kmeans": {"k": IPC, "geometry": "cosine on unit DINO embeddings",
             "initialization": "spherical k-means++", "n_init": km.KMEANS_N_INIT,
             "max_iterations": km.KMEANS_MAX_ITER, "center_update": "normalized mean",
             "representative": "maximum cosine similarity to cluster center",
             "rng_namespace": "spherical-kmeans3-v1", "total_empty_cluster_repair_events": empty, "runs": runs},
             "manifests": manifests}
    km.atomic_json(a.output_root / f"selection_audits/{a.dataset_name}.json", audit)
    print(json.dumps({"status": "complete", "arms": list(arms), "empty_cluster_events": empty}))


if __name__ == "__main__":
    main()
