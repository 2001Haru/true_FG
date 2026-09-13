"""Fully audit saved quadrant indices against the registered balanced schedule."""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import torch


def stable_offsets(paths: list[str], seed: int) -> list[int]:
    ranking = sorted(
        range(len(paths)),
        key=lambda index: hashlib.sha256(
            f"fkd-quadrant-v1\0{seed}\0{paths[index]}".encode("utf-8")
        ).digest(),
    )
    offsets = [0] * len(paths)
    for rank, index in enumerate(ranking):
        offsets[index] = rank % 4
    return offsets


def random_permutation_tile(path: str, seed: int, epoch: int) -> int:
    block, position = divmod(epoch, 4)
    permutation = sorted(
        range(4),
        key=lambda tile: hashlib.sha256(
            f"fkd-tile-random-permutation-v1\0{seed}\0{path}\0{block}\0{tile}".encode("utf-8")
        ).digest(),
    )
    return permutation[position]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fkd-dir", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--batch-size", default=20, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--schedule", choices=("cyclic", "random_permutation"), default="cyclic")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = sorted(str(path) for path in args.image_root.glob("*/*") if path.is_file())
    if len(paths) != 300:
        raise RuntimeError(f"expected 300 images, found {len(paths)}")
    offsets = stable_offsets(paths, args.seed)
    generator = torch.Generator().manual_seed(42)
    sampler = torch.utils.data.RandomSampler(range(len(paths)), generator=generator)
    per_image = [Counter() for _ in paths]
    epoch_counts = []
    payload_entries = set()
    mismatches = 0
    for epoch in range(args.epochs):
        order = list(iter(sampler))
        observed = []
        batches = (len(paths) + args.batch_size - 1) // args.batch_size
        for batch in range(batches):
            payload = torch.load(
                args.fkd_dir / f"epoch_{epoch}/batch_{batch}.tar",
                map_location="cpu", weights_only=False,
            )
            payload_entries.add(len(payload))
            if len(payload) != 7:
                raise RuntimeError(f"epoch {epoch} batch {batch}: payload length {len(payload)}")
            observed.extend(map(int, payload[6].tolist()))
        if len(observed) != len(paths):
            raise RuntimeError(f"epoch {epoch}: {len(observed)} quadrant entries")
        counts = Counter(observed)
        if args.schedule == "cyclic" and counts != Counter({0: 75, 1: 75, 2: 75, 3: 75}):
            raise RuntimeError(f"epoch {epoch}: unbalanced quadrants {counts}")
        epoch_counts.append({str(key): counts[key] for key in range(4)})
        for index, quadrant in zip(order, observed):
            expected = ((offsets[index] + epoch) % 4 if args.schedule == "cyclic"
                        else random_permutation_tile(paths[index], args.seed, epoch))
            mismatches += int(quadrant != expected)
            per_image[index][quadrant] += 1
    image_balance = [{str(key): counts[key] for key in range(4)} for counts in per_image]
    if any(counts != {"0": 100, "1": 100, "2": 100, "3": 100} for counts in image_balance):
        raise RuntimeError("at least one image does not use each quadrant exactly 100 times")
    if mismatches:
        raise RuntimeError(f"saved quadrant schedule mismatches: {mismatches}")
    result = {
        "status": "complete",
        "fkd_dir": str(args.fkd_dir.resolve()),
        "image_root": str(args.image_root.resolve()),
        "images": len(paths),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "quadrant_seed": args.seed,
        "payload_entries": sorted(payload_entries),
        "schedule": ("stable path ranking modulo 4 plus epoch modulo 4"
                     if args.schedule == "cyclic" else
                     "stable random permutation per parent path and four-epoch block"),
        "saved_schedule_mismatches": mismatches,
        "per_epoch_quadrant_counts_unique": sorted({tuple(row.values()) for row in epoch_counts}),
        "per_image_quadrant_counts_unique": sorted({tuple(row.values()) for row in image_balance}),
    }
    output = args.output or args.fkd_dir / "quadrant_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
