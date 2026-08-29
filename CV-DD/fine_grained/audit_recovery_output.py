import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image

from config import get_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Audit recovered IPC5 and sampled IPC1/3 ImageFolders")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--recovery-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_tree(root: Path, classes: int, ipc: int) -> dict:
    if not root.is_dir():
        raise RuntimeError(f"missing recovery tree: {root}")
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    expected_dirs = [f"{index:05d}" for index in range(classes)]
    if [path.name for path in class_dirs] != expected_dirs:
        raise RuntimeError(f"class directories do not match 00000..{classes - 1:05d}: {root}")

    tree_digest = hashlib.sha256()
    file_hashes = {}
    for class_id, class_dir in enumerate(class_dirs):
        files = sorted(class_dir.glob("*.jpg"))
        if len(files) != ipc:
            raise RuntimeError(
                f"class {class_id} in {root} contains {len(files)} images, expected {ipc}"
            )
        for path in files:
            relative = path.relative_to(root).as_posix()
            digest = sha256(path)
            file_hashes[relative] = digest
            tree_digest.update(relative.encode("utf-8"))
            tree_digest.update(bytes.fromhex(digest))
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if image.size != (224, 224) or image.mode != "RGB":
                    raise RuntimeError(
                        f"invalid image {path}: size={image.size}, mode={image.mode}"
                    )
    return {
        "root": str(root.resolve()),
        "classes": classes,
        "ipc": ipc,
        "files": classes * ipc,
        "tree_sha256": tree_digest.hexdigest(),
        "file_hashes": file_hashes,
    }


def main() -> None:
    args = parse_args()
    cfg = get_dataset(args.dataset_name)
    manifest_path = args.recovery_root / "recovery_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"missing recovery manifest: {manifest_path}")
    recovery_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if recovery_manifest.get("status") != "complete":
        raise RuntimeError(f"recovery manifest is not complete: {manifest_path}")

    trees = {
        str(ipc): audit_tree(args.recovery_root / f"ipc{ipc}", cfg.classes, ipc)
        for ipc in (1, 3, 5)
    }
    ipc5_hashes = trees["5"]["file_hashes"]
    for ipc in (1, 3):
        subset_hashes = trees[str(ipc)]["file_hashes"]
        for relative, digest in subset_hashes.items():
            if ipc5_hashes.get(relative) != digest:
                raise RuntimeError(f"IPC{ipc} is not a byte-identical IPC5 subset: {relative}")

    for tree in trees.values():
        tree.pop("file_hashes")
    payload = {
        "status": "complete",
        "dataset": cfg.name,
        "recovery_seed": recovery_manifest.get("recovery_seed"),
        "teacher_sha256": recovery_manifest.get("teacher_sha256"),
        "patch_tree_sha256": recovery_manifest.get("patch_tree_sha256"),
        "trees": trees,
        "sampling_relation": "IPC1 and IPC3 are byte-identical relative-path subsets of IPC5",
    }
    output = args.output or args.recovery_root / "recovery_output_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
