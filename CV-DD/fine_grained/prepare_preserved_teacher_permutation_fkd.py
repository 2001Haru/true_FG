"""Build a fixed FKD cache that permutes only unprotected Teacher logits."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff", ".webp"}


def imagefolder_targets(root, classes):
    directories = sorted(path for path in root.iterdir() if path.is_dir())
    if len(directories) != classes:
        raise RuntimeError(f"expected {classes} classes, found {len(directories)}")
    targets = []
    for class_id, directory in enumerate(directories):
        files = sorted(path for path in directory.rglob("*")
                       if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        targets.extend([class_id] * len(files))
    return targets


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--fkd-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--classes", type=int, default=100)
    args = parser.parse_args()
    targets = imagefolder_targets(args.images, args.classes)
    number_images = len(targets)
    if number_images != 300:
        raise RuntimeError(f"expected IPC3 tree with 300 images, found {number_images}")
    args.destination.mkdir(parents=True, exist_ok=True)
    sampler_generator = torch.Generator().manual_seed(args.fkd_seed)
    rows = identity_rows = tied_argmax_rows = moved_values = 0
    max_true_logit_difference = 0.0

    for epoch in range(args.epochs):
        order = torch.randperm(number_images, generator=sampler_generator).tolist()
        torch.randperm(number_images, generator=sampler_generator)[:0]
        epoch_output = args.destination / f"epoch_{epoch}"
        epoch_output.mkdir(exist_ok=True)
        for batch_index, offset in enumerate(range(0, number_images, args.batch_size)):
            source_path = args.source / f"epoch_{epoch}" / f"batch_{batch_index}.tar"
            payload = torch.load(source_path, map_location="cpu", weights_only=False)
            if len(payload) not in (6, 7) or payload[2] is not None:
                raise RuntimeError(f"expected an unmixed FKD payload at {source_path}")
            original = payload[5]
            transformed = original.clone()
            batch_order = order[offset:offset + original.shape[0]]
            for row, image_index in enumerate(batch_order):
                logits = original[row]
                true_class = targets[image_index]
                maximum = logits.max()
                argmax_ties = torch.nonzero(logits == maximum, as_tuple=False).flatten().tolist()
                if len(argmax_ties) > 1:
                    tied_argmax_rows += 1
                protected = set(argmax_ties); protected.add(true_class)
                movable = [index for index in range(args.classes) if index not in protected]
                digest = hashlib.sha256(
                    f"fkd-preserve-true-argmax-v1\0{args.seed}\0{args.source.resolve()}\0{epoch}\0{batch_index}\0{row}".encode()
                ).digest()
                generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
                permutation = torch.randperm(len(movable), generator=generator).tolist()
                if permutation == list(range(len(movable))):
                    identity_rows += 1
                indices = torch.tensor(movable, dtype=torch.long)
                transformed[row, indices] = logits[indices[permutation]]
                moved_values += sum(old != new for old, new in enumerate(permutation))
                rows += 1
                max_true_logit_difference = max(
                    max_true_logit_difference,
                    abs(float(transformed[row, true_class]) - float(logits[true_class])),
                )
                if int(transformed[row].argmax()) != int(logits.argmax()):
                    raise RuntimeError("Teacher argmax was not preserved")
                if not torch.equal(transformed[row].sort().values, logits.sort().values):
                    raise RuntimeError("logit multiset was not preserved")
            payload[5] = transformed
            torch.save(payload, epoch_output / f"batch_{batch_index}.tar")

    result = {
        "status": "complete", "transformation": "preserve_true_and_teacher_argmax_permute_other_logits",
        "source_fkd": str(args.source.resolve()), "destination_fkd": str(args.destination.resolve()),
        "images": str(args.images.resolve()), "seed": args.seed, "fkd_seed": args.fkd_seed,
        "epochs": args.epochs, "batch_size": args.batch_size, "classes": args.classes,
        "views": rows, "identity_rows": identity_rows, "argmax_tie_rows": tied_argmax_rows,
        "moved_coordinate_values": moved_values,
        "max_true_logit_absolute_difference": max_true_logit_difference,
        "invariants": ["true-class raw logit", "Teacher argmax class", "complete raw-logit multiset",
                       "T20 probability multiset", "entropy", "probability-vector norm"],
        "randomness": "stable SHA-256 per source-FKD/epoch/batch/row; no global RNG mutation",
    }
    output = args.destination / "permutation_manifest.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
