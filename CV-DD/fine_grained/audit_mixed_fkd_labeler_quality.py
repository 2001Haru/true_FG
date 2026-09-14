"""Compare labelers on the same sampled RRC+CutMix FKD views."""

import argparse
import json
from pathlib import Path

import torch

from vit_teacher_common import atomic_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff", ".webp"}


def summary(values):
    tensor = torch.cat(values).double()
    return {
        "count": tensor.numel(),
        "mean": float(tensor.mean()),
        "sample_sd": float(tensor.std(unbiased=True)),
        "p10": float(tensor.quantile(0.10)),
        "median": float(tensor.quantile(0.50)),
        "p90": float(tensor.quantile(0.90)),
    }


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--fkd", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--sample-epochs", nargs="+", type=int, default=(0, 200, 399))
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch-size", default=20, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    class_dirs = sorted(path for path in args.images.iterdir() if path.is_dir())
    targets = []
    for class_id, directory in enumerate(class_dirs):
        files = sorted(
            path for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        targets.extend([class_id] * len(files))
    if len(class_dirs) != 100 or len(targets) != 300:
        raise RuntimeError(f"unexpected image tree: classes={len(class_dirs)}, images={len(targets)}")
    requested = set(args.sample_epochs)
    generator = torch.Generator().manual_seed(args.seed)
    orders = {}
    for epoch in range(max(requested) + 1):
        order = torch.randperm(len(targets), generator=generator).tolist()
        torch.randperm(len(targets), generator=generator)[:0]
        if epoch in requested:
            orders[epoch] = order
    roots = {name: Path(path) for name, path in args.fkd}
    values = {
        name: {key: [] for key in (
            "argmax_is_receiver", "argmax_is_donor", "argmax_is_either",
            "top2_contains_both", "mixed_target_ce_T1", "mixed_target_ce_T20",
            "area_weighted_target_probability_T1", "entropy_T1", "entropy_T20",
            "logit_std",
        )}
        for name in roots
    }
    pairwise = {
        f"{left}_vs_{right}": {"argmax_agreement": [], "symmetric_kl_T20": []}
        for left, right in zip(list(roots)[:-1], list(roots)[1:])
    }
    for epoch in sorted(requested):
        order = orders[epoch]
        for batch_index, offset in enumerate(range(0, len(targets), args.batch_size)):
            batch_indices = order[offset : offset + args.batch_size]
            receiver = torch.tensor([targets[index] for index in batch_indices], dtype=torch.long)
            payloads = {
                name: torch.load(root / f"epoch_{epoch}" / f"batch_{batch_index}.tar", map_location="cpu", weights_only=False)
                for name, root in roots.items()
            }
            reference = next(iter(payloads.values()))
            mix_index = reference[2].long()
            donor = receiver[mix_index]
            x1, y1, x2, y2 = (int(value) for value in reference[4])
            donor_weight = ((x2 - x1) * (y2 - y1)) / float(224 * 224)
            receiver_weight = 1.0 - donor_weight
            distributions = {}
            for name, payload in payloads.items():
                if not torch.equal(payload[2].long(), mix_index) or list(map(int, payload[4])) != [x1, y1, x2, y2]:
                    raise RuntimeError(f"metadata mismatch at epoch{epoch}/batch{batch_index}: {name}")
                logits = payload[5].float()
                logp1 = logits.log_softmax(1)
                p1 = logp1.exp()
                logp20 = (logits / 20.0).log_softmax(1)
                p20 = logp20.exp()
                distributions[name] = (p1, p20)
                prediction = logits.argmax(1)
                top2 = logits.topk(2, dim=1).indices
                row = torch.arange(logits.shape[0])
                both = top2.eq(receiver[:, None]).any(1) & top2.eq(donor[:, None]).any(1)
                target_ce1 = -(receiver_weight * logp1[row, receiver] + donor_weight * logp1[row, donor])
                target_ce20 = -(receiver_weight * logp20[row, receiver] + donor_weight * logp20[row, donor])
                weighted_probability = receiver_weight * p1[row, receiver] + donor_weight * p1[row, donor]
                current = values[name]
                current["argmax_is_receiver"].append(prediction.eq(receiver).double())
                current["argmax_is_donor"].append(prediction.eq(donor).double())
                current["argmax_is_either"].append((prediction.eq(receiver) | prediction.eq(donor)).double())
                current["top2_contains_both"].append(both.double())
                current["mixed_target_ce_T1"].append(target_ce1)
                current["mixed_target_ce_T20"].append(target_ce20)
                current["area_weighted_target_probability_T1"].append(weighted_probability)
                current["entropy_T1"].append((-(p1 * logp1).sum(1)))
                current["entropy_T20"].append((-(p20 * logp20).sum(1)))
                current["logit_std"].append(logits.std(1, unbiased=False))
            names = list(roots)
            for left, right in zip(names[:-1], names[1:]):
                p_left, p20_left = distributions[left]
                p_right, p20_right = distributions[right]
                pair = pairwise[f"{left}_vs_{right}"]
                pair["argmax_agreement"].append(p_left.argmax(1).eq(p_right.argmax(1)).double())
                kl_lr = (p20_left * (p20_left.clamp_min(1e-12).log() - p20_right.clamp_min(1e-12).log())).sum(1)
                kl_rl = (p20_right * (p20_right.clamp_min(1e-12).log() - p20_left.clamp_min(1e-12).log())).sum(1)
                pair["symmetric_kl_T20"].append(0.5 * (kl_lr + kl_rl))
    result = {
        "status": "complete",
        "sample_epochs": sorted(requested),
        "sampled_views_per_labeler": len(requested) * len(targets),
        "labelers": {name: {key: summary(chunks) for key, chunks in metrics.items()} for name, metrics in values.items()},
        "pairwise": {name: {key: summary(chunks) for key, chunks in metrics.items()} for name, metrics in pairwise.items()},
    }
    atomic_json(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
