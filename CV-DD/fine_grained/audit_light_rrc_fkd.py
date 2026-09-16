"""Audit the frozen light-square-RRC trajectory and an exact replay FKD."""

import argparse
import json
import math
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    areas, sides = [], []
    mismatches = {"coords": 0, "flip": 0, "mix_index": 0, "mix_lambda": 0, "mix_bbox": 0}
    batches = examples = 0
    for epoch in range(400):
        for batch in range(15):
            ref = torch.load(args.reference / f"epoch_{epoch}/batch_{batch}.tar", map_location="cpu", weights_only=False)
            cand = torch.load(args.candidate / f"epoch_{epoch}/batch_{batch}.tar", map_location="cpu", weights_only=False)
            for index, key in enumerate(mismatches):
                left, right = ref[index], cand[index]
                equal = torch.equal(left, right) if torch.is_tensor(left) else left == right
                mismatches[key] += int(not equal)
            coords = ref[0].float()
            height, width = coords[:, 2], coords[:, 3]
            areas.extend((height * width).tolist())
            sides.extend(height.tolist())
            if float((height - width).abs().max()) > 1e-7:
                raise RuntimeError(f"non-square crop epoch={epoch} batch={batch}")
            batches += 1; examples += len(coords)
    ref_manifest = json.loads((args.reference / "relabel_manifest.json").read_text())
    result = {
        "status": "complete" if not sum(mismatches.values()) else "failed",
        "reference": str(args.reference.resolve()), "candidate": str(args.candidate.resolve()),
        "batches": batches, "examples": examples, "mismatch_batches": mismatches,
        "requested_scale": [0.8, 1.0], "requested_ratio": [1.0, 1.0],
        "manifest_scale": [ref_manifest["min_scale_crops"], ref_manifest["max_scale_crops"]],
        "manifest_ratio": [ref_manifest["min_aspect_ratio_crops"], ref_manifest["max_aspect_ratio_crops"]],
        "actual_area": {"min": min(areas), "mean": sum(areas) / len(areas), "max": max(areas)},
        "actual_side_pixels": {"min": 224 * min(sides), "mean": 224 * sum(sides) / len(sides),
                               "max": 224 * max(sides)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "complete" or result["manifest_scale"] != [0.8, 1.0] or result["manifest_ratio"] != [1.0, 1.0]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
