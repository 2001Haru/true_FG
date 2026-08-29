import argparse
import hashlib
import json
import os
import re
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


def audit_tree(root: Path, classes: int, ipc: int, expected_class_names=None) -> dict:
    if not root.is_dir():
        raise RuntimeError(f"missing recovery tree: {root}")
    class_dirs = [path for path in root.iterdir() if path.is_dir()]
    parsed = []
    for path in class_dirs:
        match = re.fullmatch(r"(?:new)?(\d+)", path.name)
        if match is None:
            raise RuntimeError(f"unrecognized class directory {path.name!r}: {root}")
        parsed.append((int(match.group(1)), path))
    parsed.sort()
    if [class_id for class_id, _ in parsed] != list(range(classes)):
        raise RuntimeError(f"class directory IDs do not cover 0..{classes - 1}: {root}")
    class_dirs = [path for _, path in parsed]
    class_names = [path.name for path in class_dirs]
    if expected_class_names is not None and class_names != expected_class_names:
        raise RuntimeError(f"class directory names differ from IPC5: {root}")

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
        "class_names": class_names,
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

    ipc5 = audit_tree(args.recovery_root / "ipc5", cfg.classes, 5)
    class_names = ipc5["class_names"]
    trees = {
        "1": audit_tree(args.recovery_root / "ipc1", cfg.classes, 1, class_names),
        "3": audit_tree(args.recovery_root / "ipc3", cfg.classes, 3, class_names),
        "5": ipc5,
    }
    ipc5_hashes = trees["5"]["file_hashes"]
    for ipc in (1, 3):
        subset_hashes = trees[str(ipc)]["file_hashes"]
        for relative, digest in subset_hashes.items():
            if ipc5_hashes.get(relative) != digest:
                raise RuntimeError(f"IPC{ipc} is not a byte-identical IPC5 subset: {relative}")

    for tree in trees.values():
        tree.pop("file_hashes")
        tree.pop("class_names")
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
