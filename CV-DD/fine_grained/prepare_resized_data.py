import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from config import get_dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def atomic_json_dump(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def image_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def resize_one(source: Path, target: Path, skip_completed: bool) -> tuple[str, bool]:
    if skip_completed and target.is_file():
        with Image.open(target) as existing:
            if existing.size == (224, 224):
                return "224x224", True
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        source_size = f"{image.width}x{image.height}"
        image_format = image.format or ("PNG" if target.suffix.lower() == ".png" else "JPEG")
        resized = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    temporary = target.with_name(target.name + ".tmp")
    save_options = {}
    if image_format.upper() in {"JPEG", "JPG"}:
        image_format = "JPEG"
        # High quality avoids adding unnecessary compression artifacts. The
        # choice is recorded in the manifest for later sensitivity checks.
        save_options.update(quality=95, optimize=False)
    resized.save(temporary, format=image_format, **save_options)
    os.replace(temporary, target)
    return source_size, False


def main() -> None:
    parser = argparse.ArgumentParser("Create immutable physical 224x224 fine-grained ImageFolders")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    cfg = get_dataset(args.dataset_name)
    records = {}
    total_written = 0
    total_skipped = 0

    for split in ("train", "test"):
        source_split = args.source_dir / split
        output_split = args.output_dir / split
        classes = sorted(path for path in source_split.iterdir() if path.is_dir())
        if len(classes) != cfg.classes:
            raise RuntimeError(f"{source_split}: found {len(classes)} classes, expected {cfg.classes}")
        files = image_files(source_split)

        def task(source: Path) -> tuple[str, bool]:
            relative = source.relative_to(source_split)
            return resize_one(source, output_split / relative, args.skip_completed)

        source_sizes = Counter()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for index, (source_size, skipped) in enumerate(executor.map(task, files), start=1):
                source_sizes[source_size] += 1
                total_skipped += int(skipped)
                total_written += int(not skipped)
                if index % 1000 == 0 or index == len(files):
                    print(
                        f"{cfg.name} {split}: {index}/{len(files)} "
                        f"written={total_written} skipped={total_skipped}",
                        flush=True,
                    )

        output_files = image_files(output_split)
        if len(output_files) != len(files):
            raise RuntimeError(f"{split}: output count {len(output_files)} != source count {len(files)}")
        invalid = []
        for path in output_files:
            with Image.open(path) as image:
                if image.size != (224, 224):
                    invalid.append(str(path))
        if invalid:
            raise RuntimeError(f"{split}: {len(invalid)} output images are not 224x224")
        records[split] = {
            "images": len(files),
            "classes": len(classes),
            "source_unique_sizes": len(source_sizes),
            "source_size_counts": dict(source_sizes),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(
        {
            "status": "complete",
            "dataset": cfg.to_dict(),
            "source_dir": str(args.source_dir.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "target_size": [224, 224],
            "interpolation": "PIL bilinear",
            "jpeg_quality": 95,
            "written": total_written,
            "skipped": total_skipped,
            "splits": records,
        },
        args.output_dir / "resize_manifest.json",
    )
    print(f"Physical resize complete: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
