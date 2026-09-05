"""Add immutable shell-random IPC1 selections to a frozen DINO geometry audit."""

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


SEEDS = (0, 1, 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_key(seed: int, relative_path: str) -> bytes:
    return hashlib.sha256(
        f"shell-random-v1\0{seed}\0{relative_path}".encode("utf-8")
    ).digest()


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


def tree_sha256(records: list[dict], selection_root: Path) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda row: row["selected_path"]):
        relative = Path(record["selected_path"]).relative_to(selection_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(bytes.fromhex(record["source_sha256"]))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-audit", required=True, type=Path)
    parser.add_argument("--selection-base", required=True, type=Path)
    parser.add_argument("--extension-audit", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--classes", required=True, type=int)
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    args = parser.parse_args()

    parent_bytes = args.geometry_audit.read_bytes()
    parent_hash = hashlib.sha256(parent_bytes).hexdigest()
    geometry = json.loads(parent_bytes)
    expected = {
        "status": "complete",
        "dataset": args.dataset_name,
        "classes": args.classes,
        "ipc": 1,
        "source_images": len(geometry.get("images", [])),
    }
    for key, value in expected.items():
        if geometry.get(key) != value:
            raise RuntimeError(
                f"geometry audit {key}={geometry.get(key)!r}, expected {value!r}"
            )
    shell_definition = geometry.get("shell", {})
    if shell_definition.get("prototype_correctness_filter") is not False:
        raise RuntimeError("shell-random requires the revised pure-radial shell")
    if (
        shell_definition.get("radial_percentile_low_inclusive") != 0.70
        or shell_definition.get("radial_percentile_high_inclusive") != 0.95
    ):
        raise RuntimeError("shell-random requires the frozen 70%-95% shell")

    by_class = defaultdict(list)
    for row in geometry["images"]:
        by_class[int(row["class_id"])].append(row)
    if sorted(by_class) != list(range(args.classes)):
        raise RuntimeError("geometry audit does not cover every class")
    selected_by_seed = {}
    extension_rows = []
    for seed in SEEDS:
        arm = f"shell_random_rseed{seed}"
        selection_root = args.selection_base / arm
        selected = []
        for class_id in range(args.classes):
            candidates = [row for row in by_class[class_id] if row["in_edge_shell"]]
            if not candidates:
                raise RuntimeError(f"class {class_id} has an empty frozen radial shell")
            source_root = Path(geometry["source_root"])
            chosen = min(
                candidates,
                key=lambda row: selection_key(
                    seed, Path(row["source_path"]).relative_to(source_root).as_posix()
                ),
            )
            source = Path(chosen["source_path"])
            destination = selection_root / chosen["class_folder"] / source.name
            materialize(source, destination, args.link_mode)
            record = {
                "class_id": class_id,
                "class_folder": chosen["class_folder"],
                "source_path": str(source.resolve()),
                "selected_path": str(destination.absolute()),
                "source_sha256": sha256(source),
                "own_centroid_similarity": chosen["own_centroid_similarity"],
                "radial_cosine_distance": chosen["radial_cosine_distance"],
                "nearest_rival_similarity": chosen["nearest_rival_similarity"],
                "nearest_rival_class_id": chosen["nearest_rival_class_id"],
                "nearest_rival_class_folder": chosen["nearest_rival_class_folder"],
                "prototype_margin": chosen["prototype_margin"],
                "within_class_radial_percentile": chosen[
                    "within_class_radial_percentile"
                ],
                "in_edge_shell": True,
                "existing_selection_overlap": chosen.get("selected_by", []),
            }
            selected.append(record)
            extension_rows.append({"selection_seed": seed, **record})
        selected_by_seed[seed] = (selection_root, selected)

    overlap_counts = {
        str(seed): {
            method: sum(
                method in row["existing_selection_overlap"]
                for row in selected_by_seed[seed][1]
            )
            for method in (
                "centroid",
                "rival_facing_edge",
                "outward_edge",
                "edge_high_margin",
            )
        }
        for seed in SEEDS
    }
    extension = {
        "status": "complete",
        "experiment": "dino_sixarm_ipc1",
        "selection_method": "shell_random",
        "dataset": args.dataset_name,
        "classes": args.classes,
        "ipc": 1,
        "selection_seeds": list(SEEDS),
        "selection_algorithm": (
            "uniform pseudo-random choice within each frozen shell by minimum "
            "SHA256('shell-random-v1' + NUL + seed + NUL + source-relative-path)"
        ),
        "shell_definition": "0.70 <= within-class percentile(r) <= 0.95",
        "uses_rival_similarity": False,
        "geometry_audit": str(args.geometry_audit.resolve()),
        "geometry_audit_sha256": parent_hash,
        "dino_model_sha256": geometry["dino_model_sha256"],
        "dino_preprocessor_sha256": geometry["dino_preprocessor_sha256"],
        "source_root": geometry["source_root"],
        "overlap_counts_with_existing_arms": overlap_counts,
        "images": extension_rows,
    }
    atomic_json(args.extension_audit, extension)
    if hashlib.sha256(args.geometry_audit.read_bytes()).hexdigest() != parent_hash:
        raise RuntimeError("parent geometry audit changed while building extension")

    for seed, (selection_root, records) in selected_by_seed.items():
        arm = f"shell_random_rseed{seed}"
        actual = {
            path.relative_to(selection_root).as_posix()
            for class_dir in selection_root.iterdir()
            if class_dir.is_dir()
            for path in class_dir.iterdir()
            if path.is_file()
        }
        expected_paths = {
            Path(row["selected_path"]).relative_to(selection_root).as_posix()
            for row in records
        }
        if actual != expected_paths or len(actual) != args.classes:
            raise RuntimeError(f"selected ImageFolder audit failed for {arm}")
        manifest = {
            "status": "complete",
            "experiment": "dino_sixarm_ipc1",
            "dataset": args.dataset_name,
            "classes": args.classes,
            "ipc": 1,
            "selection_method": "shell_random",
            "selection_arm": arm,
            "selection_seed": seed,
            "selection_root": str(selection_root.resolve()),
            "selection_images": args.classes,
            "selection_audit": str(args.extension_audit.resolve()),
            "selection_audit_sha256": sha256(args.extension_audit),
            "parent_geometry_audit": str(args.geometry_audit.resolve()),
            "parent_geometry_audit_sha256": parent_hash,
            "selected_tree_sha256": tree_sha256(records, selection_root),
            "materialization": args.link_mode,
            "images": records,
        }
        atomic_json(args.selection_base / "manifests" / f"{arm}.json", manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "dataset": args.dataset_name,
                "arms": [f"shell_random_rseed{seed}" for seed in SEEDS],
                "parent_geometry_audit_sha256": parent_hash,
                "overlap_counts_with_existing_arms": overlap_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

