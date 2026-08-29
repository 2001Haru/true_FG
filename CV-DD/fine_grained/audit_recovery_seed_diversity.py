import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from config import get_dataset


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (224, 224):
            raise RuntimeError(f"unexpected size {image.size}: {path}")
        return np.asarray(image, dtype=np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser("Quantify pixel diversity across recovery seeds")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--recovery-parent", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=(41, 42, 43))
    parser.add_argument("--ipc", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = get_dataset(args.dataset_name)
    roots = {
        seed: args.recovery_parent / f"rseed{seed}" / f"ipc{args.ipc}"
        for seed in args.seeds
    }
    relative_paths = {}
    for seed, root in roots.items():
        paths = sorted(path.relative_to(root) for path in root.glob("*/*.jpg"))
        if len(paths) != cfg.classes * args.ipc:
            raise RuntimeError(f"seed {seed}: {len(paths)} files != {cfg.classes * args.ipc}")
        relative_paths[seed] = paths
    reference = relative_paths[args.seeds[0]]
    for seed in args.seeds[1:]:
        if relative_paths[seed] != reference:
            raise RuntimeError(f"relative paths differ for seed {seed}")

    pairs = []
    for left_seed, right_seed in itertools.combinations(args.seeds, 2):
        maes = []
        rmses = []
        exact = 0
        for relative in reference:
            left = load_rgb(roots[left_seed] / relative)
            right = load_rgb(roots[right_seed] / relative)
            difference = left - right
            mae = float(np.mean(np.abs(difference)))
            rmse = float(np.sqrt(np.mean(np.square(difference))))
            maes.append(mae)
            rmses.append(rmse)
            exact += int(np.array_equal(left, right))
        mean_rmse = float(np.mean(rmses))
        pairs.append({
            "left_seed": left_seed,
            "right_seed": right_seed,
            "images": len(reference),
            "exact_duplicates": exact,
            "mean_mae_0_1": float(np.mean(maes)),
            "median_mae_0_1": float(np.median(maes)),
            "minimum_mae_0_1": float(np.min(maes)),
            "maximum_mae_0_1": float(np.max(maes)),
            "mean_rmse_0_1": mean_rmse,
            "psnr_from_mean_rmse_db": (
                float("inf") if mean_rmse == 0 else 20.0 * math.log10(1.0 / mean_rmse)
            ),
        })
    payload = {
        "status": "complete",
        "dataset": cfg.name,
        "ipc": args.ipc,
        "seeds": args.seeds,
        "relative_paths": len(reference),
        "pairs": pairs,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
