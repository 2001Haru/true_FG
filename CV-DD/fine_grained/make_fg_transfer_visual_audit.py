"""Render source, stored strip, inverse decode, and F1 decode for FG transfer packs."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFont

from paired_source_dataset import PairedSourceIndex


def annotate(image, text, box=None):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    if box is not None:
        draw.rectangle(tuple(round(v) for v in box), outline=(0, 255, 0), width=2)
    draw.rectangle((0, 0, 223, 20), fill=(0, 0, 0))
    draw.text((4, 4), text, fill=(255, 255, 255), font=ImageFont.load_default())
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=8)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text())
    plain = PairedSourceIndex(args.manifest, "compressed")
    fir = PairedSourceIndex(args.manifest, "compressed_fir")
    axis = payload["compression_axis"]
    candidates = []
    for parent in payload["records"]:
        for source in parent["sources"]:
            candidates.append((float(source["subject_density"]), parent["parent_index"],
                               source["subsource_index"]))
    candidates.sort()
    indices = np.linspace(0, len(candidates) - 1, args.rows).round().astype(int)
    chosen = [candidates[index] for index in indices]
    canvas = Image.new("RGB", (5 * 224, args.rows * 224), "white")
    records = []
    for row_index, (subject_density, parent_index, source_index) in enumerate(chosen):
        parent = payload["records"][parent_index]
        source = parent["sources"][source_index]
        original = Image.open(source["reference_path"]).convert("RGB")
        packed = Image.open(parent["packed_path"]).convert("RGB")
        offset, size = int(source["encoded_top"]), int(source["encoded_height"])
        if axis == "row":
            strip = packed.crop((0, offset, 224, offset + size))
        else:
            strip = packed.crop((offset, 0, offset + size, 224))
        strip = strip.resize((224, 224), Image.Resampling.NEAREST)
        decoded = plain.load(parent_index, source_index)
        enhanced = fir.load(parent_index, source_index)
        difference = ImageEnhance.Brightness(ImageChops.difference(enhanced, decoded)).enhance(5)
        box = source["official_box224_xyxy"]
        panels = (
            annotate(original, f"original sd={subject_density:.3f}", box),
            annotate(strip, f"stored {size}px {axis}"),
            annotate(decoded, "inverse decode", box),
            annotate(enhanced, "F1 decode", box),
            annotate(difference, "5x |F1-BBox|"),
        )
        for column, panel in enumerate(panels):
            canvas.paste(panel, (column * 224, row_index * 224))
        records.append({
            "parent": parent_index,
            "subsource": source_index,
            "identity": source["identity"],
            "subject_density": subject_density,
            "background_density": source["background_density"],
            "bbox_axis_pixels": source["bbox_axis_pixels"],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    audit = {
        "status": "complete",
        "dataset": payload["dataset"],
        "axis": axis,
        "sampling": "even quantiles of subject density",
        "columns": ["original+bbox", "stored strip", "inverse decode", "F1 decode", "5x abs difference"],
        "rows": records,
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
