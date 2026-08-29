import argparse
import json
import math
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Audit an FKD label tree before post-evaluation")
    parser.add_argument("--fkd-dir", required=True, type=Path)
    parser.add_argument("--images", required=True, type=int)
    parser.add_argument("--classes", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(f"FKD audit failed: {message}")


def main() -> None:
    args = parse_args()
    if min(args.images, args.classes, args.batch_size, args.epochs) <= 0:
        fail("images, classes, batch-size, and epochs must be positive")
    if not args.fkd_dir.is_dir():
        fail(f"missing directory: {args.fkd_dir}")

    batches_per_epoch = math.ceil(args.images / args.batch_size)
    expected_names = {f"batch_{index}.tar" for index in range(batches_per_epoch)}
    total_bytes = 0
    for epoch in range(args.epochs):
        epoch_dir = args.fkd_dir / f"epoch_{epoch}"
        if not epoch_dir.is_dir():
            fail(f"missing epoch directory: {epoch_dir}")
        files = list(epoch_dir.glob("batch_*.tar"))
        observed_names = {path.name for path in files}
        if observed_names != expected_names:
            missing = sorted(expected_names - observed_names)
            extra = sorted(observed_names - expected_names)
            fail(f"epoch {epoch}: missing={missing[:5]} extra={extra[:5]}")
        empty = [path.name for path in files if path.stat().st_size <= 0]
        if empty:
            fail(f"epoch {epoch}: empty batch files={empty[:5]}")
        total_bytes += sum(path.stat().st_size for path in files)

    sample_epochs = sorted({0, args.epochs // 2, args.epochs - 1})
    sample_batches = sorted({0, batches_per_epoch - 1})
    samples = []
    for epoch in sample_epochs:
        for batch_index in sample_batches:
            path = args.fkd_dir / f"epoch_{epoch}" / f"batch_{batch_index}.tar"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(payload, (list, tuple)) or len(payload) != 6:
                fail(f"unexpected payload at {path}: expected six entries")
            soft_labels = payload[5]
            if not isinstance(soft_labels, torch.Tensor) or soft_labels.ndim != 2:
                fail(f"soft labels at {path} are not a rank-2 tensor")
            expected_rows = (
                args.batch_size
                if batch_index < batches_per_epoch - 1
                else args.images - args.batch_size * (batches_per_epoch - 1)
            )
            if tuple(soft_labels.shape) != (expected_rows, args.classes):
                fail(
                    f"soft-label shape at {path} is {tuple(soft_labels.shape)}, "
                    f"expected {(expected_rows, args.classes)}"
                )
            if not torch.isfinite(soft_labels).all():
                fail(f"non-finite soft labels at {path}")
            samples.append({
                "epoch": epoch,
                "batch": batch_index,
                "shape": list(soft_labels.shape),
                "dtype": str(soft_labels.dtype),
            })

    output = args.output or args.fkd_dir / "fkd_audit.json"
    result = {
        "status": "complete",
        "fkd_dir": str(args.fkd_dir.resolve()),
        "images": args.images,
        "classes": args.classes,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "batches_per_epoch": batches_per_epoch,
        "batch_files": args.epochs * batches_per_epoch,
        "total_bytes": total_bytes,
        "content_samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
