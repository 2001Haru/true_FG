"""Build balanced same-source 2x2 mosaics from ImageNette A/B shared 112 caches."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, ipc: int, identity: str) -> bytes:
    return hashlib.sha256(
        f"imagenette-same-source-mosaic-v1\0{seed}\0{ipc}\0{identity}".encode("utf-8")
    ).digest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ab-construction-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--construction-seed", default=42, type=int)
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    manifest_path = args.output_root / "construction_manifest.json"
    if args.skip_completed and manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            print(f"ImageNette same-source mosaics already complete: {manifest_path}")
            return
    ab = json.loads(args.ab_construction_manifest.read_text(encoding="utf-8"))
    if ab.get("status") != "complete" or ab.get("selection_temperature") != 20.0:
        raise RuntimeError("invalid ImageNette H20 A/B construction manifest")
    records = {}
    for ipc in (10, 50):
        ipc_records = []
        for class_record in ab["class_records"]:
            sources = class_record["images"][:ipc]
            if len(sources) != ipc:
                raise RuntimeError(f"class {class_record['class_id']} IPC{ipc}: source count")
            permutation = sorted(
                range(ipc),
                key=lambda index: stable_key(
                    args.construction_seed, ipc, sources[index]["sample_id"]
                ),
            )
            output_class = args.output_root / f"ipc{ipc}" / class_record["class_folder"]
            output_class.mkdir(parents=True, exist_ok=True)
            mosaics = []
            occurrences = {source_id: [] for source_id in range(ipc)}
            for image_id in range(ipc):
                canvas = Image.new("RGB", (224, 224))
                quadrants = []
                for quadrant in range(4):
                    source_id = permutation[(image_id + quadrant) % ipc]
                    cache = Path(sources[source_id]["cache112_path"])
                    with Image.open(cache) as image:
                        patch = image.convert("RGB")
                    if patch.size != (112, 112):
                        raise RuntimeError(f"invalid cache size: {cache} {patch.size}")
                    row, column = divmod(quadrant, 2)
                    canvas.paste(patch, (column * 112, row * 112))
                    occurrences[source_id].append(quadrant)
                    quadrants.append(
                        {
                            "quadrant": quadrant,
                            "source_id": source_id,
                            "sample_id": sources[source_id]["sample_id"],
                            "cache112_path": str(cache.resolve()),
                            "cache112_sha256": sources[source_id]["cache112_sha256"],
                        }
                    )
                output = output_class / f"class{class_record['class_id']:05d}_id{image_id:05d}.jpg"
                canvas.save(output, format="JPEG")
                mosaics.append(
                    {"image_id": image_id, "quadrants": quadrants,
                     "output_path": str(output.resolve()), "output_sha256": sha256(output)}
                )
            if any(sorted(values) != [0, 1, 2, 3] for values in occurrences.values()):
                raise RuntimeError(f"class {class_record['class_id']} IPC{ipc}: unbalanced occurrences")
            ipc_records.append(
                {
                    "class_id": class_record["class_id"],
                    "class_folder": class_record["class_folder"],
                    "source_permutation": permutation,
                    "sources": sources,
                    "mosaics": mosaics,
                    "source_quadrant_occurrences": {
                        str(source_id): values for source_id, values in occurrences.items()
                    },
                }
            )
        files = list((args.output_root / f"ipc{ipc}").glob("*/*.jpg"))
        if len(files) != 10 * ipc:
            raise RuntimeError(f"IPC{ipc}: found {len(files)} images")
        records[str(ipc)] = ipc_records
    payload = {
        "status": "complete", "dataset": "imagenet-nette", "ipcs": [10, 50],
        "construction_seed": args.construction_seed,
        "source": "exact H20 A manifest sources and exact shared 112x112 PNG caches",
        "ab_construction_manifest": str(args.ab_construction_manifest.resolve()),
        "ab_construction_manifest_sha256": sha256(args.ab_construction_manifest),
        "rule": "stable per-class permutation; mosaic t quadrant q uses permutation[(t+q) mod IPC]",
        "output_images_per_class": "IPC",
        "source_occurrences": 4,
        "each_source_quadrants": [0, 1, 2, 3],
        "jpeg": "PIL defaults",
        "records": records,
    }
    atomic_json(payload, manifest_path)
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
