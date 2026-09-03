"""Audit a fine-grained CoDA ImageFolder and write an atomic manifest."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image

from config import get_dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--synthetic-dir", required=True, type=Path)
    parser.add_argument("--ipc", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--generation-config", required=True, type=Path)
    args = parser.parse_args()
    cfg = get_dataset(args.dataset_name)
    expected_classes = sorted(path.name for path in (args.data_dir / "train").iterdir() if path.is_dir())
    actual_classes = sorted(path.name for path in args.synthetic_dir.iterdir() if path.is_dir())
    if len(expected_classes) != cfg.classes or actual_classes != expected_classes:
        raise RuntimeError("CoDA class folders do not exactly match the prepared training ImageFolder")
    if not args.generation_config.is_file():
        raise FileNotFoundError(args.generation_config)

    tree_digest = hashlib.sha256()
    files = 0
    for class_name in actual_classes:
        class_dir = args.synthetic_dir / class_name
        images = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(images) != args.ipc:
            raise RuntimeError(f"{class_name} contains {len(images)} images, expected IPC={args.ipc}")
        for path in images:
            relative = path.relative_to(args.synthetic_dir).as_posix()
            digest = sha256(path)
            tree_digest.update(relative.encode("utf-8"))
            tree_digest.update(bytes.fromhex(digest))
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if image.mode != "RGB" or image.size != (224, 224):
                    raise RuntimeError(f"Invalid CoDA image {path}: mode={image.mode}, size={image.size}")
            files += 1
    payload = {
        "status": "complete",
        "method": "CoDA fine-grained adapter",
        "dataset": cfg.name,
        "classes": cfg.classes,
        "ipc": args.ipc,
        "files": files,
        "image_mode": "RGB",
        "image_size": [224, 224],
        "synthetic_dir": str(args.synthetic_dir.resolve()),
        "tree_sha256": tree_digest.hexdigest(),
        "generation_config": str(args.generation_config.resolve()),
        "generation_config_sha256": sha256(args.generation_config),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
