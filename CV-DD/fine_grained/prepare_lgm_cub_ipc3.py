"""Safely extract, class-map, and audit the external CUB LGM IPC3 archive."""

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--reference-train", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    raw_root = args.output_root / "raw"
    selected_root = args.output_root / "selected"
    reference_classes = sorted(path for path in args.reference_train.iterdir() if path.is_dir())
    if len(reference_classes) != 200:
        raise RuntimeError(f"expected 200 reference classes, found {len(reference_classes)}")
    by_prefix = {}
    for path in reference_classes:
        prefix = path.name.split(".", 1)[0]
        if len(prefix) != 3 or not prefix.isdigit() or prefix in by_prefix:
            raise RuntimeError(f"invalid reference class folder: {path.name}")
        by_prefix[prefix] = path.name

    with zipfile.ZipFile(args.archive) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 600:
            raise RuntimeError(f"archive must contain exactly 600 files, found {len(members)}")
        for item in members:
            parts = PurePosixPath(item.filename).parts
            if len(parts) != 3 or parts[0] != "CUB200_LGM_global_IPC3":
                raise RuntimeError(f"unexpected archive member: {item.filename}")
            if parts[1] not in by_prefix or Path(parts[2]).suffix.lower() != ".png":
                raise RuntimeError(f"invalid class/image member: {item.filename}")
            destination = (raw_root / Path(*parts)).resolve()
            if raw_root.resolve() not in destination.parents:
                raise RuntimeError(f"archive path traversal: {item.filename}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                with archive.open(item) as source, destination.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)

    records = []
    counts = {}
    modes, sizes = set(), set()
    raw_dataset = raw_root / "CUB200_LGM_global_IPC3"
    for prefix in sorted(by_prefix):
        sources = sorted((raw_dataset / prefix).glob("*.png"))
        if len(sources) != 3:
            raise RuntimeError(f"class {prefix}: expected 3 PNGs, found {len(sources)}")
        class_name = by_prefix[prefix]
        destination_class = selected_root / class_name
        destination_class.mkdir(parents=True, exist_ok=True)
        counts[class_name] = len(sources)
        for source in sources:
            with Image.open(source) as image:
                image.verify()
            with Image.open(source) as image:
                modes.add(image.mode); sizes.add(tuple(image.size))
            destination = destination_class / source.name
            if destination.exists() or destination.is_symlink():
                if not destination.is_symlink() or destination.resolve() != source.resolve():
                    raise RuntimeError(f"canonical collision: {destination}")
            else:
                os.symlink(source.resolve(), destination)
            records.append({"archive_class": prefix, "canonical_class": class_name,
                            "source": str(source.resolve()), "selected": str(destination.absolute()),
                            "sha256": sha256(source)})
    actual = sorted(path for path in selected_root.glob("*/*") if path.is_file())
    if len(actual) != 600 or set(counts.values()) != {3}:
        raise RuntimeError("canonical ImageFolder count audit failed")
    payload = {"status": "complete", "dataset": "CUB_imsize224", "method": "LGM_global",
               "ipc": 3, "classes": 200, "images": 600,
               "archive": str(args.archive.resolve()), "archive_sha256": sha256(args.archive),
               "raw_root": str(raw_root.resolve()), "selected_root": str(selected_root.resolve()),
               "class_mapping": "three-digit archive prefix mapped to identical prefix of canonical CUB semantic folder",
               "class_counts": counts, "image_modes": sorted(modes), "image_sizes": sorted(map(list, sizes)),
               "records": records}
    atomic_json(payload, args.output_root / "lgm_input_manifest.json")
    print(json.dumps({key: value for key, value in payload.items() if key not in ("class_counts", "records")}, indent=2))


if __name__ == "__main__":
    main()
