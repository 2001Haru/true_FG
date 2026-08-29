import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from config import get_dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def image_stats(path_text: str) -> tuple[np.ndarray, np.ndarray, int]:
    path = Path(path_text)
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (224, 224):
            raise RuntimeError(f"not 224x224: {path} ({image.size})")
        pixels = np.asarray(image, dtype=np.float64).reshape(-1, 3) / 255.0
    return pixels.sum(axis=0), np.square(pixels).sum(axis=0), pixels.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser("Compute exact channel statistics for a prepared dataset split")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = get_dataset(args.dataset_name)
    split_dir = args.data_dir / args.split
    paths = sorted(
        path for path in split_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"no images found: {split_dir}")

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    pixels = 0
    with Pool(processes=args.workers) as pool:
        for partial_sum, partial_square_sum, partial_pixels in pool.imap_unordered(
            image_stats, map(str, paths), chunksize=16
        ):
            channel_sum += partial_sum
            channel_square_sum += partial_square_sum
            pixels += partial_pixels
    mean = channel_sum / pixels
    variance = np.maximum(channel_square_sum / pixels - np.square(mean), 0.0)
    std = np.sqrt(variance)
    expected_mean = np.asarray(cfg.mean)
    expected_std = np.asarray(cfg.std)
    payload = {
        "status": "complete",
        "dataset": cfg.name,
        "split": args.split,
        "data_dir": str(args.data_dir.resolve()),
        "images": len(paths),
        "pixels": pixels,
        "mean": mean.tolist(),
        "std_population": std.tolist(),
        "official_mean": list(cfg.mean),
        "official_std": list(cfg.std),
        "mean_delta": (mean - expected_mean).tolist(),
        "std_delta": (std - expected_std).tolist(),
        "max_abs_mean_delta": float(np.max(np.abs(mean - expected_mean))),
        "max_abs_std_delta": float(np.max(np.abs(std - expected_std))),
        "statistic_definition": "population statistics over all decoded RGB pixels scaled to [0,1]",
    }
    output = args.output or args.data_dir / f"{args.split}_channel_stats.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
