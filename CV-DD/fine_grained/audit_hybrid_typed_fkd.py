"""Audit that hybrid FKD changes only slot0 RRC coordinates to full-frame."""
import argparse
import json
import os
from pathlib import Path

import torch


def equal(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.equal(torch.as_tensor(a), torch.as_tensor(b))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--images", default=300, type=int)
    parser.add_argument("--batch-size", default=20, type=int)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    generator = torch.Generator().manual_seed(args.seed)
    index_dataset = torch.utils.data.TensorDataset(torch.arange(args.images))
    sampler = torch.utils.data.RandomSampler(index_dataset, generator=generator)
    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=args.batch_size, drop_last=False)
    full = torch.tensor([0.0, 0.0, 1.0, 1.0])
    counts = {"views": 0, "full_views": 0, "mosaic_views": 0, "full_not_full": 0,
              "mosaic_coords_changed": 0, "flip_mismatch": 0, "mix_index_mismatch": 0,
              "mix_lambda_mismatch": 0, "mix_bbox_mismatch": 0}
    for epoch in range(args.epochs):
        for batch_index, indices in enumerate(batch_sampler):
            source = torch.load(args.source / f"epoch_{epoch}/batch_{batch_index}.tar", weights_only=False)
            candidate = torch.load(args.candidate / f"epoch_{epoch}/batch_{batch_index}.tar", weights_only=False)
            for position, index in enumerate(indices):
                counts["views"] += 1
                if index % 3 == 0:
                    counts["full_views"] += 1
                    counts["full_not_full"] += int(not torch.allclose(torch.as_tensor(candidate[0][position]).float(), full, rtol=0, atol=1e-7))
                else:
                    counts["mosaic_views"] += 1
                    counts["mosaic_coords_changed"] += int(not equal(source[0][position], candidate[0][position]))
                counts["flip_mismatch"] += int(not equal(source[1][position], candidate[1][position]))
            counts["mix_index_mismatch"] += int(not equal(source[2], candidate[2]))
            counts["mix_lambda_mismatch"] += int(not equal(source[3], candidate[3]))
            counts["mix_bbox_mismatch"] += int(not equal(source[4], candidate[4]))
    errors = {key: value for key, value in counts.items() if key.endswith(("mismatch", "changed", "not_full")) and value}
    result = {"status": "complete" if not errors else "failed", "source": str(args.source),
              "candidate": str(args.candidate), "slot_rule": "sorted ImageFolder index modulo 3: slot0 full, slots1/2 mosaics",
              "counts": counts, "errors": errors}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".json.tmp"); tmp.write_text(json.dumps(result, indent=2) + "\n"); os.replace(tmp, args.output)
    print(json.dumps(result, indent=2))
    if errors:
        raise RuntimeError(errors)


if __name__ == "__main__":
    main()
