"""Gate reuse of A-view FKD logits while training on paired B-view images."""

import argparse
import json
import os
from pathlib import Path

import torch


def equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    if isinstance(left, float) or isinstance(right, float):
        return float(left) == float(right)
    return left == right


def image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*/*") if path.is_file() or path.is_symlink())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-images", required=True, type=Path)
    parser.add_argument("--b-images", required=True, type=Path)
    parser.add_argument("--construction-manifest", required=True, type=Path)
    parser.add_argument("--a-fkd", required=True, type=Path)
    parser.add_argument("--b-fkd", required=True, type=Path)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--batches-per-epoch", default=25, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    construction = json.loads(args.construction_manifest.read_text(encoding="utf-8"))
    if construction.get("status") != "complete":
        raise RuntimeError("construction manifest is not complete")
    a_files = image_files(args.a_images)
    b_files = image_files(args.b_images)
    if len(a_files) != 500 or len(b_files) != 500:
        raise RuntimeError(f"image counts A/B: {len(a_files)}/{len(b_files)}")
    expected_a = []
    expected_b = []
    for class_record in construction["class_records"]:
        expected_a.extend(Path(row["source_path"]).resolve() for row in class_record["sources"])
        expected_b.extend(Path(row["downsample_up_path"]).resolve() for row in class_record["sources"])
    a_resolved = [path.resolve() for path in a_files]
    b_resolved = [path.resolve() for path in b_files]
    if a_resolved != expected_a:
        raise RuntimeError("A ImageFolder order does not match construction source order")
    if b_resolved != expected_b:
        raise RuntimeError("B ImageFolder order does not match construction source order")

    field_names = ("rrc_coords", "flip", "cutmix_index", "cutmix_lambda", "cutmix_bbox")
    mismatches = {name: 0 for name in field_names}
    logits_shape_mismatches = 0
    payload_length_mismatches = 0
    compared_batches = 0
    compared_examples = 0
    for epoch in range(args.epochs):
        for batch in range(args.batches_per_epoch):
            a_path = args.a_fkd / f"epoch_{epoch}/batch_{batch}.tar"
            b_path = args.b_fkd / f"epoch_{epoch}/batch_{batch}.tar"
            a = torch.load(a_path, map_location="cpu", weights_only=False)
            b = torch.load(b_path, map_location="cpu", weights_only=False)
            if len(a) != 6 or len(b) != 6:
                payload_length_mismatches += 1
                continue
            for index, name in enumerate(field_names):
                mismatches[name] += int(not equal(a[index], b[index]))
            logits_shape_mismatches += int(a[5].shape != b[5].shape)
            compared_batches += 1
            compared_examples += int(a[5].shape[0])
    total_mismatches = sum(mismatches.values()) + logits_shape_mismatches + payload_length_mismatches
    if compared_batches != args.epochs * args.batches_per_epoch or compared_examples != args.epochs * 500:
        raise RuntimeError("incomplete FKD comparison")
    result = {
        "status": "complete" if total_mismatches == 0 else "failed",
        "a_images": str(args.a_images.resolve()),
        "b_images": str(args.b_images.resolve()),
        "a_fkd": str(args.a_fkd.resolve()),
        "b_fkd": str(args.b_fkd.resolve()),
        "samples_aligned": 500,
        "imagefolder_index_mapping": "exact one-to-one",
        "epochs": args.epochs,
        "batches_per_epoch": args.batches_per_epoch,
        "compared_batches": compared_batches,
        "compared_examples": compared_examples,
        "metadata_mismatch_batches": mismatches,
        "logits_shape_mismatches": logits_shape_mismatches,
        "payload_length_mismatches": payload_length_mismatches,
        "total_mismatches": total_mismatches,
        "interpretation": "A logits can supervise B iff status is complete; logits themselves are intentionally not compared",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
