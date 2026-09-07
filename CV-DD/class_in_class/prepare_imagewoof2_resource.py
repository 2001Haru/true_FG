"""Build an audited ImageWoof2 train/test view from the legacy image pool.

The fastai CSV defines the current 70/30 split.  The legacy local resource has
the same 12,954 image payloads but a 12,454/500 split and appends the synset to
each filename.  This script verifies a bijection and creates symlinks only; it
does not copy, rename, or modify source images.
"""

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def canonical_source_name(path):
    suffix = "_" + path.parent.name
    stem = path.stem
    if not stem.endswith(suffix):
        raise RuntimeError(f"legacy filename lacks expected synset suffix: {path}")
    return stem[: -len(suffix)] + path.suffix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", required=True, type=Path)
    parser.add_argument("--official-csv", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    legacy = args.legacy_root.resolve()
    csv_path = args.official_csv.resolve()
    output = args.output_root.resolve()
    staging = output.with_name(output.name + ".building")
    if output.exists() or staging.exists():
        raise RuntimeError(f"refusing existing output/staging: {output} / {staging}")

    source_index = {}
    for legacy_split in ("train", "val"):
        for path in sorted((legacy / legacy_split).glob("*/*")):
            if path.suffix.lower() not in EXTENSIONS:
                continue
            key = (path.parent.name, canonical_source_name(path))
            if key in source_index:
                raise RuntimeError(f"duplicate canonical source key: {key}")
            source_index[key] = path.resolve()
    if len(source_index) != 12954:
        raise RuntimeError(f"legacy image count is {len(source_index)}, expected 12954")

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    if len(rows) != 12954:
        raise RuntimeError(f"official CSV rows={len(rows)}, expected 12954")
    planned = []
    seen = set()
    split_counts = Counter()
    class_counts = defaultdict(Counter)
    source_split_crosswalk = Counter()
    for row in rows:
        csv_split, synset, filename = row["path"].split("/")
        if row["noisy_labels_0"] != synset:
            raise RuntimeError(f"clean label/path mismatch: {row['path']}")
        expected_valid = csv_split == "val"
        if (row["is_valid"] == "True") != expected_valid:
            raise RuntimeError(f"is_valid/path mismatch: {row['path']}")
        target_split = "test" if expected_valid else "train"
        key = (synset, filename)
        if key in seen:
            raise RuntimeError(f"duplicate official image identity: {key}")
        seen.add(key)
        source = source_index.get(key)
        if source is None:
            raise RuntimeError(f"missing legacy payload: {row['path']}")
        planned.append((target_split, synset, filename, source))
        split_counts[target_split] += 1
        class_counts[target_split][synset] += 1
        source_split_crosswalk[(target_split, source.parts[-3])] += 1
    if set(source_index) != seen:
        raise RuntimeError("official CSV and legacy source are not a bijection")
    if split_counts != Counter({"train": 9025, "test": 3929}):
        raise RuntimeError(f"unexpected ImageWoof2 split counts: {split_counts}")

    staging.mkdir(parents=True)
    file_rows = []
    tree = hashlib.sha256()
    try:
        for index, (split, synset, filename, source) in enumerate(sorted(planned)):
            destination_dir = staging / split / synset
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / filename
            destination.symlink_to(source)
            with Image.open(source) as image:
                image.verify()
            digest = sha256(source)
            relative = f"{split}/{synset}/{filename}"
            tree.update(relative.encode("utf-8")); tree.update(b"\0")
            tree.update(digest.encode("ascii")); tree.update(b"\n")
            file_rows.append({
                "relative_path": relative, "source_path": str(source),
                "source_sha256": digest, "bytes": source.stat().st_size,
            })
            if index % 1000 == 0:
                print(f"verified {index + 1}/{len(planned)}", flush=True)
        manifest = {
            "status": "complete", "dataset": "fastai ImageWoof2 full-size clean labels",
            "construction": "official CSV 70/30 split over a bijective legacy payload pool; absolute symlinks",
            "official_csv": str(csv_path), "official_csv_sha256": sha256(csv_path),
            "legacy_root": str(legacy), "output_root": str(output),
            "split_counts": dict(split_counts),
            "per_class_counts": {split: dict(sorted(counts.items())) for split, counts in class_counts.items()},
            "class_to_idx": {name: index for index, name in enumerate(sorted(class_counts["train"]))},
            "source_split_crosswalk": {f"{target}_from_{source}": value for (target, source), value in sorted(source_split_crosswalk.items())},
            "tree_sha256": tree.hexdigest(), "files": file_rows,
        }
        atomic_json(staging / "resource_manifest.json", manifest)
        staging.rename(output)
    except Exception:
        # Preserve staging for forensic inspection; never delete source or a
        # partially constructed resource automatically.
        raise
    print(json.dumps({key: manifest[key] for key in ("status", "split_counts", "per_class_counts", "class_to_idx", "tree_sha256")}, indent=2))


if __name__ == "__main__":
    main()
