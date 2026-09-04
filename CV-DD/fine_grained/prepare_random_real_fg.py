"""Build and audit deterministic nested random-real ImageFolder subsets."""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_key(seed: int, relative_path: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{relative_path}".encode("utf-8")).digest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--classes", required=True, type=int)
    parser.add_argument("--ipc", required=True, type=int)
    parser.add_argument("--selection-seed", required=True, type=int)
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    args = parser.parse_args()

    source_root = (args.data_dir / "train").resolve()
    class_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
    if len(class_dirs) != args.classes:
        raise RuntimeError(f"Source class count {len(class_dirs)} != {args.classes}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    expected_destinations = set()
    tree_digest = hashlib.sha256()
    for class_id, class_dir in enumerate(class_dirs):
        candidates = sorted(
            path.resolve()
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(candidates) < args.ipc:
            raise RuntimeError(
                f"Class {class_dir.name} has {len(candidates)} images, below IPC={args.ipc}"
            )
        ranked = sorted(
            candidates,
            key=lambda path: selection_key(
                args.selection_seed, path.relative_to(source_root).as_posix()
            ),
        )
        destination_class = args.output_dir / class_dir.name
        destination_class.mkdir(exist_ok=True)
        for rank, source in enumerate(ranked[: args.ipc]):
            destination = destination_class / source.name
            relative_destination = destination.relative_to(args.output_dir).as_posix()
            expected_destinations.add(relative_destination)
            if destination.exists() or destination.is_symlink():
                if args.link_mode == "symlink":
                    if not destination.is_symlink() or destination.resolve() != source:
                        raise RuntimeError(f"Selection destination collision: {destination}")
                elif destination.is_symlink() or sha256(destination) != sha256(source):
                    raise RuntimeError(f"Selection destination collision: {destination}")
            else:
                if args.link_mode == "symlink":
                    os.symlink(source, destination)
                else:
                    shutil.copy2(source, destination)
            content_hash = sha256(source)
            tree_digest.update(relative_destination.encode("utf-8"))
            tree_digest.update(bytes.fromhex(content_hash))
            records.append(
                {
                    "class_id": class_id,
                    "class_folder": class_dir.name,
                    "rank": rank,
                    "source_path": str(source),
                    "selected_path": str(destination.absolute()),
                    "source_sha256": content_hash,
                }
            )

    actual_classes = sorted(path for path in args.output_dir.iterdir() if path.is_dir())
    if [path.name for path in actual_classes] != [path.name for path in class_dirs]:
        raise RuntimeError("Selected ImageFolder class directories differ from source")
    actual_destinations = {
        path.relative_to(args.output_dir).as_posix()
        for class_dir in actual_classes
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if actual_destinations != expected_destinations:
        raise RuntimeError("Selected ImageFolder contains missing or extra image links")
    payload = {
        "status": "complete",
        "method": "random_real",
        "dataset": args.dataset_name,
        "classes": args.classes,
        "ipc": args.ipc,
        "selection_seed": args.selection_seed,
        "selection_algorithm": "SHA256(seed + NUL + source_relative_path), ascending",
        "materialization": args.link_mode,
        "nested_ipc_property": "IPC1 subset IPC3 subset IPC5 for the same seed",
        "source_root": str(source_root),
        "selected_root": str(args.output_dir.resolve()),
        "selected_images": len(records),
        "selected_tree_sha256": tree_digest.hexdigest(),
        "images": records,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.manifest)
    print(json.dumps({key: payload[key] for key in payload if key != "images"}, indent=2))


if __name__ == "__main__":
    main()
