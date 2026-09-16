"""Pack six nested RandomReal sources per class into three two-strip parent slots."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_aircraft_retarget_compression import bbox_density, encode_vertical_area, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection-manifest", required=True, type=Path)
    p.add_argument("--raw-images", required=True, type=Path)
    p.add_argument("--boxes", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--schedule-seed", type=int, default=42)
    a = p.parse_args()
    selection = json.loads(a.selection_manifest.read_text())
    if selection.get("status") != "complete" or selection.get("ipc") != 10 or selection.get("selection_seed") != 0:
        raise RuntimeError("requires completed nested RandomReal rseed0 IPC10 manifest")
    boxes = {}
    for line in a.boxes.read_text().splitlines():
        z = line.split(); boxes[z[0]] = tuple(map(int, z[1:5]))
    by_class = {}
    for row in selection["images"]: by_class.setdefault(row["class_folder"], []).append(row)
    classes = sorted(by_class); records = []; density_reduced = 0
    for class_id, class_name in enumerate(classes):
        chosen = sorted(by_class[class_name], key=lambda row: row["rank"])[:6]
        if len(chosen) != 6 or len({Path(row["source_path"]).name for row in chosen}) != 6:
            raise RuntimeError(f"class {class_name} lacks six unique nested sources")
        for slot, pair in enumerate(((0, 3), (1, 4), (2, 5))):
            packed = Image.new("RGB", (224, 224)); sources = []
            for subsource, rank_index in enumerate(pair):
                selected = chosen[rank_index]; reference = Path(selected["source_path"])
                identity = reference.stem; raw_path = a.raw_images / f"{identity}.jpg"
                if not reference.is_file() or not raw_path.is_file() or identity not in boxes:
                    raise RuntimeError(f"missing source/raw/bbox {identity}")
                with Image.open(reference) as handle: image = handle.convert("RGB")
                with Image.open(raw_path) as handle: raw_size = handle.size
                if image.size != (224, 224): raise RuntimeError(f"reference not 224: {reference}")
                xmin, ymin, xmax, ymax = boxes[identity]
                y0 = (ymin - 1.0) / raw_size[1] * 224.0; y1 = ymax / raw_size[1] * 224.0
                density, height, reduced = bbox_density(y0, y1); density_reduced += int(reduced)
                encoded, _ = encode_vertical_area(np.asarray(image, dtype=np.uint8), density)
                packed.paste(Image.fromarray(encoded, "RGB"), (0, subsource * 112))
                sources.append({"subsource_index": subsource, "rank": int(selected["rank"]),
                                "identity": identity, "reference_path": str(reference.resolve()),
                                "raw_path": str(raw_path.resolve()), "raw_size": list(raw_size),
                                "official_bbox_1indexed": [xmin, ymin, xmax, ymax], "bbox_rows": height,
                                "density_reduced": bool(reduced), "density": density.tolist()})
            destination = a.output_root / "packed" / class_name / f"parent_{slot}.png"
            destination.parent.mkdir(parents=True, exist_ok=True); temp = destination.with_suffix(".png.tmp")
            packed.save(temp, format="PNG", compress_level=0); os.replace(temp, destination)
            records.append({"parent_index": class_id * 3 + slot, "class_id": class_id, "class": class_name,
                            "slot": slot, "packed_path": str(destination.resolve()),
                            "packed_sha256": sha256(destination), "sources": sources})
    if len(records) != 300: raise RuntimeError(len(records))
    output = {"status": "complete", "protocol": "aircraft_ipc3_six_source_bbox_pack_v1",
              "classes": classes, "class_count": 100, "storage_ipc": 3, "parents": 300,
              "sources_per_parent": 2, "sources_per_class": 6, "unique_sources": 600,
              "pairing": "nested ranks (0,3), (1,4), (2,5)", "schedule_seed": a.schedule_seed,
              "schedule": "stable parent offset plus epoch modulo 2; adjacent epochs exhaust both sources",
              "encoding": "each source bbox-retarget area-encoded to 224x112 uint8 PNG strip; two strips stacked",
              "decoding": "select recorded strip then inverse-map with its frozen 224-row density before augmentation",
              "density_reduced_sources": density_reduced, "records": records}
    out = a.output_root / "six_source_manifest.json"; out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp"); tmp.write_text(json.dumps(output, indent=2) + "\n"); os.replace(tmp, out)
    print(json.dumps({k: output[k] for k in ("status", "parents", "unique_sources", "density_reduced_sources")}, indent=2))


if __name__ == "__main__": main()
