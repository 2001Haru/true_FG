"""Pack k complete sources per storage slot using a fixed axis and phi budget."""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_aircraft_retarget_compression import encode_vertical_area


def sha(path):
    digest = hashlib.sha256(); digest.update(path.read_bytes()); return digest.hexdigest()


def strip_sizes(k):
    base, remainder = divmod(224, k)
    return [base + int(index < remainder) for index in range(k)]


def density(box, axis, target, background_minimum):
    lo, hi = ((box[1], box[3]) if axis == "row" else (box[0], box[2]))
    first = max(0, min(223, math.floor(lo)))
    last = max(first + 1, min(224, math.ceil(hi)))
    subject = np.zeros(224, dtype=bool); subject[first:last] = True
    occupied, background = int(subject.sum()), int((~subject).sum())
    subject_density = min(.9, (target - background_minimum * background) / occupied)
    background_density = ((target - subject_density * occupied) / background
                          if background else target / 224)
    values = np.where(subject, subject_density, background_density).astype(np.float64)
    values[-1] += float(target) - float(values.sum())
    if values.min() <= 0 or abs(float(values.sum()) - target) > 1e-9:
        raise RuntimeError((values.min(), values.sum(), target, occupied, background_minimum))
    return values, occupied, subject_density, background_density


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--bbox-manifest", required=True, type=Path)
    parser.add_argument("--axis", choices=("row", "column"), required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--storage-ipc", type=int, required=True)
    parser.add_argument("--phi", type=float, default=.106)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--mean", nargs=3, type=float, required=True)
    parser.add_argument("--std", nargs=3, type=float, required=True)
    parser.add_argument("--schedule-seed", type=int, default=42)
    args = parser.parse_args()
    if args.k < 2 or args.storage_ipc < 1 or not 0 < args.phi < 1:
        raise ValueError((args.k, args.storage_ipc, args.phi))
    selection = json.loads(args.selection_manifest.read_text())
    bbox = json.loads(args.bbox_manifest.read_text())
    boxes = {row["prepared_path"]: row for row in bbox["rows"]}
    by_class = {}
    for row in selection["images"]:
        by_class.setdefault(row["class_folder"], []).append(row)
    axis_key = "height_fraction" if args.axis == "row" else "width_fraction"
    selected_count = args.k * args.storage_ipc
    # h-bar is a dataset-level geometry statistic, fixed before source selection.
    mean_axis_fraction = float(np.mean([row[axis_key] for row in bbox["rows"]]))
    background_minimum = args.phi / (args.k * (1.0 - mean_axis_fraction))
    sizes = strip_sizes(args.k)
    groups = [tuple(slot + source * args.storage_ipc for source in range(args.k))
              for slot in range(args.storage_ipc)]
    records, density_stats = [], []
    for class_id, class_name in enumerate(sorted(by_class)):
        chosen = sorted(by_class[class_name], key=lambda item: item["rank"])[:selected_count]
        if len(chosen) != selected_count or len({item["source_path"] for item in chosen}) != selected_count:
            raise RuntimeError((class_name, len(chosen), selected_count))
        for slot, ranks in enumerate(groups):
            packed = Image.new("RGB", (224, 224)); sources = []; offset = 0
            for subsource, (rank, target) in enumerate(zip(ranks, sizes)):
                selected = chosen[rank]; reference = Path(selected["source_path"]).resolve()
                meta = boxes[str(reference)]
                image = np.asarray(Image.open(reference).convert("RGB"), np.uint8)
                values, span, subject_density, background_density = density(
                    meta["box224_xyxy"], args.axis, target, background_minimum)
                oriented = image if args.axis == "row" else image.transpose(1, 0, 2)
                encoded, _ = encode_vertical_area(oriented, values)
                tile = Image.fromarray(encoded if args.axis == "row" else encoded.transpose(1, 0, 2))
                packed.paste(tile, (0, offset) if args.axis == "row" else (offset, 0))
                sources.append({
                    "subsource_index": subsource, "rank": int(selected["rank"]),
                    "identity": reference.stem, "reference_path": str(reference),
                    "prepared_relative": meta["prepared_relative"], "raw_size": meta["raw_size"],
                    "official_box224_xyxy": meta["box224_xyxy"], "bbox_axis_pixels": span,
                    "density": values.tolist(), "subject_density": subject_density,
                    "background_density": background_density, "compression_axis": args.axis,
                    "encoded_top": offset, "encoded_height": target,
                })
                offset += target; density_stats.append((subject_density, background_density))
            destination = args.output_root / "packed" / class_name / f"parent_{slot}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".tmp")
            packed.save(temporary, format="PNG", compress_level=0); os.replace(temporary, destination)
            records.append({"parent_index": class_id * args.storage_ipc + slot,
                            "class_id": class_id, "class": class_name, "slot": slot,
                            "packed_path": str(destination.resolve()),
                            "packed_sha256": sha(destination), "sources": sources})
    values = np.asarray(density_stats)
    output = {
        "status": "complete", "protocol": "fg_closed_form_axis_k_pack_v1",
        "dataset": selection["dataset"], "classes": sorted(by_class),
        "class_count": len(by_class), "storage_ipc": args.storage_ipc,
        "parents": len(records), "sources_per_parent": args.k,
        "sources_per_class": selected_count, "unique_sources": len(by_class) * selected_count,
        "pairing": str(groups), "strip_heights": sizes, "schedule_seed": args.schedule_seed,
        "schedule": f"stable parent offset plus epoch modulo {args.k}",
        "compression_axis": args.axis, "phi": args.phi,
        "mean_selected_axis_box_fraction": mean_axis_fraction,
        "background_minimum": background_minimum,
        "background_formula": "phi/(k*(1-mean selected-axis box fraction))",
        "normalization_mean": args.mean, "normalization_std": args.std,
        "subject_density_stats": {"mean": float(values[:, 0].mean()),
                                  "p10": float(np.quantile(values[:, 0], .1)),
                                  "median": float(np.median(values[:, 0])),
                                  "min": float(values[:, 0].min()), "max": float(values[:, 0].max())},
        "background_density_stats": {"mean": float(values[:, 1].mean()),
                                     "min": float(values[:, 1].min()), "max": float(values[:, 1].max())},
        "records": records,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "base_manifest.json").write_text(json.dumps(output, indent=2) + "\n")
    # A strict R0 control has the same parents, source-index schedule, flips and
    # CutMix trajectory, but every scheduled subsource resolves to rank zero.
    anchor = copy.deepcopy(output); anchor["protocol"] += "+r0_anchor_control"
    anchor["unique_sources"] = len(by_class); anchor["sources_per_class"] = 1
    for record in anchor["records"]:
        first = copy.deepcopy(record["sources"][0])
        record["sources"] = []
        for subsource in range(args.k):
            duplicate = copy.deepcopy(first); duplicate["subsource_index"] = subsource
            record["sources"].append(duplicate)
    (args.output_root / "anchor_manifest.json").write_text(json.dumps(anchor, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in (
        "dataset", "storage_ipc", "parents", "sources_per_parent", "unique_sources",
        "compression_axis", "phi", "mean_selected_axis_box_fraction", "background_minimum",
        "subject_density_stats")}, indent=2))


if __name__ == "__main__":
    main()
