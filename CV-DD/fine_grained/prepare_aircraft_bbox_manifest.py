"""Freeze all Aircraft official boxes in the prepared 224-warp coordinate system."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "min": float(values.min()),
            "p10": float(np.quantile(values, .1)), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9)), "max": float(values.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-train", required=True, type=Path)
    parser.add_argument("--raw-images", required=True, type=Path)
    parser.add_argument("--boxes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    boxes = {parts[0]: tuple(map(float, parts[1:5]))
             for line in args.boxes.read_text().splitlines() if (parts := line.split())}
    rows = []
    for prepared in sorted(args.prepared_train.glob("*/*")):
        identity = prepared.stem; raw = args.raw_images / f"{identity}.jpg"
        if identity not in boxes or not raw.is_file():
            raise RuntimeError((identity, raw))
        with Image.open(raw) as image:
            width, height = image.size
        xmin, ymin, xmax, ymax = boxes[identity]
        # Official Aircraft boxes are inclusive and one-indexed.
        box = [(xmin - 1) * 224 / width, (ymin - 1) * 224 / height,
               xmax * 224 / width, ymax * 224 / height]
        rows.append({
            "prepared_path": str(prepared.resolve()),
            "prepared_relative": str(prepared.relative_to(args.prepared_train)),
            "raw_path": str(raw.resolve()), "raw_size": [width, height],
            "official_box_raw_xyxy_1indexed": [xmin, ymin, xmax, ymax],
            "box224_xyxy": box,
            "width_fraction": (xmax - xmin + 1) / width,
            "height_fraction": (ymax - ymin + 1) / height,
        })
    widths = [row["width_fraction"] for row in rows]
    heights = [row["height_fraction"] for row in rows]
    output = {"status": "complete", "dataset": "A_imsize224", "images": len(rows),
              "box_stats": {"width_fraction": summary(widths),
                            "height_fraction": summary(heights),
                            "height_greater_than_width_fraction": float(
                                np.mean(np.asarray(heights) > np.asarray(widths)))},
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": "complete", "images": len(rows),
                      "box_stats": output["box_stats"]}, indent=2))


if __name__ == "__main__":
    main()
