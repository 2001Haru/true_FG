"""Verify that rescored FKD preserves every source geometry/mixing field exactly."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fkd", required=True, type=Path)
    parser.add_argument("--target-fkd", required=True, type=Path)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--batches-per-epoch", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    names = ("rrc_coords", "flip", "cutmix_index", "cutmix_lambda", "cutmix_bbox")
    mismatches = {name: 0 for name in names}
    shape_mismatches = 0
    examples = 0
    for epoch in range(args.epochs):
        for batch in range(args.batches_per_epoch):
            source = torch.load(
                args.source_fkd / f"epoch_{epoch}/batch_{batch}.tar",
                map_location="cpu", weights_only=False,
            )
            target = torch.load(
                args.target_fkd / f"epoch_{epoch}/batch_{batch}.tar",
                map_location="cpu", weights_only=False,
            )
            if len(source) != 6 or len(target) != 6:
                raise RuntimeError(f"invalid payload length at epoch {epoch} batch {batch}")
            for index, name in enumerate(names):
                mismatches[name] += int(not equal(source[index], target[index]))
            shape_mismatches += int(source[5].shape != target[5].shape)
            examples += int(target[5].shape[0])
    total = sum(mismatches.values()) + shape_mismatches
    result = {
        "status": "complete" if total == 0 else "failed",
        "source_fkd": str(args.source_fkd.resolve()),
        "target_fkd": str(args.target_fkd.resolve()),
        "epochs": args.epochs, "batches_per_epoch": args.batches_per_epoch,
        "batch_size": args.batch_size,
        "compared_batches": args.epochs * args.batches_per_epoch,
        "compared_examples": examples,
        "metadata_mismatch_batches": mismatches,
        "logit_shape_mismatches": shape_mismatches,
        "total_mismatches": total,
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
