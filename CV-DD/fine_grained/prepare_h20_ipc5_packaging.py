"""Create deterministic downsample and balanced 2x2 packaging arms from H20 IPC5."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def source_files(class_dir: Path) -> list[Path]:
    return sorted(
        path.resolve() for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    manifest_path = args.output_root / "construction_manifest.json"
    if args.skip_completed and manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            print(f"Packaging construction already complete: {manifest_path}")
            return
    selection = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if selection.get("status") != "complete" or selection.get("dataset") != "A_imsize224":
        raise RuntimeError("invalid H20 selection manifest")
    classes = sorted(path for path in args.source_root.iterdir() if path.is_dir())
    if len(classes) != 100:
        raise RuntimeError(f"found {len(classes)} source classes")
    records = []
    for class_id, class_dir in enumerate(classes):
        sources = source_files(class_dir)
        if len(sources) != 5:
            raise RuntimeError(f"{class_dir}: expected five source images, found {len(sources)}")
        cache_dir = args.output_root / "cache112" / class_dir.name
        downsample_dir = args.output_root / "downsample_up" / class_dir.name
        mosaic_dir = args.output_root / "same_source_mosaic" / class_dir.name
        cache_dir.mkdir(parents=True, exist_ok=True)
        downsample_dir.mkdir(parents=True, exist_ok=True)
        mosaic_dir.mkdir(parents=True, exist_ok=True)
        lows = []
        source_rows = []
        for source_id, source in enumerate(sources):
            with Image.open(source) as image:
                rgb = image.convert("RGB")
                if rgb.size != (224, 224):
                    raise RuntimeError(f"source is not 224x224: {source} {rgb.size}")
                low = rgb.resize((112, 112), Image.Resampling.BILINEAR)
            cache = cache_dir / f"source{source_id:02d}.png"
            low.save(cache, format="PNG")
            with Image.open(cache) as cached:
                low = cached.convert("RGB")
                upsampled = low.resize((224, 224), Image.Resampling.BILINEAR)
            downsampled = downsample_dir / f"class{class_id:05d}_id{source_id:05d}.jpg"
            upsampled.save(downsampled, format="JPEG")
            lows.append(low.copy())
            source_rows.append(
                {
                    "source_id": source_id,
                    "source_path": str(source),
                    "source_sha256": sha256(source),
                    "cache112_path": str(cache.resolve()),
                    "cache112_sha256": sha256(cache),
                    "downsample_up_path": str(downsampled.resolve()),
                    "downsample_up_sha256": sha256(downsampled),
                }
            )
        mosaics = []
        for omitted in range(5):
            canvas = Image.new("RGB", (224, 224))
            quadrants = []
            for quadrant in range(4):
                source_id = (omitted + quadrant + 1) % 5
                row, column = divmod(quadrant, 2)
                canvas.paste(lows[source_id], (column * 112, row * 112))
                quadrants.append(
                    {
                        "quadrant": quadrant,
                        "source_id": source_id,
                        "cache112_sha256": source_rows[source_id]["cache112_sha256"],
                    }
                )
            output = mosaic_dir / f"class{class_id:05d}_id{omitted:05d}.jpg"
            canvas.save(output, format="JPEG")
            mosaics.append(
                {
                    "image_id": omitted,
                    "omitted_source_id": omitted,
                    "quadrants": quadrants,
                    "output_path": str(output.resolve()),
                    "output_sha256": sha256(output),
                }
            )
        source_occurrences = {source_id: [] for source_id in range(5)}
        for mosaic in mosaics:
            for quadrant in mosaic["quadrants"]:
                source_occurrences[quadrant["source_id"]].append(quadrant["quadrant"])
        if any(sorted(values) != [0, 1, 2, 3] for values in source_occurrences.values()):
            raise RuntimeError(f"unbalanced source/quadrant schedule in class {class_id}")
        records.append(
            {
                "class_id": class_id,
                "class_folder": class_dir.name,
                "sources": source_rows,
                "mosaics": mosaics,
                "source_quadrant_occurrences": {str(key): value for key, value in source_occurrences.items()},
            }
        )
    for arm in ("downsample_up", "same_source_mosaic"):
        files = list((args.output_root / arm).glob("*/*.jpg"))
        if len(files) != 500:
            raise RuntimeError(f"{arm}: found {len(files)} outputs")
    caches = list((args.output_root / "cache112").glob("*/*.png"))
    if len(caches) != 500:
        raise RuntimeError(f"found {len(caches)} cached images")
    payload = {
        "status": "complete",
        "experiment": "H20 IPC5 fixed-source resolution and packaging construction",
        "dataset": "A_imsize224",
        "ipc": 5,
        "source_method": "entropy_t20",
        "source_root": str(args.source_root.resolve()),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": sha256(args.source_manifest),
        "classes": 100,
        "source_images": 500,
        "cache": {"size": [112, 112], "format": "lossless PNG", "interpolation": "PIL bilinear"},
        "downsample_up": {"images": 500, "upsample_size": [224, 224], "interpolation": "PIL bilinear", "jpeg": "PIL defaults"},
        "same_source_mosaic": {
            "images": 500, "size": [224, 224], "jpeg": "PIL defaults",
            "rule": "mosaic omitted=o; quadrant q receives source (o+q+1) mod 5",
            "each_source_occurrences": 4, "each_source_quadrants": [0, 1, 2, 3],
        },
        "class_records": records,
    }
    atomic_json(payload, manifest_path)
    print(json.dumps({key: value for key, value in payload.items() if key != "class_records"}, indent=2))


if __name__ == "__main__":
    main()
