import argparse
import json
import os
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from config import get_dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def resize_shorter(image: Image.Image, shorter: int) -> Image.Image:
    width, height = image.size
    if width <= height:
        size = (shorter, int(shorter * height / width))
    else:
        size = (int(shorter * width / height), shorter)
    return image.resize(size, Image.Resampling.BILINEAR)


def apply_decode_transform(image: Image.Image, mode: str) -> Image.Image:
    if mode == "none":
        return image
    if mode == "warp224":
        return image.resize((224, 224), Image.Resampling.BILINEAR)
    shorter = 224 if mode == "shorter224_centercrop224" else 256
    image = resize_shorter(image, shorter)
    width, height = image.size
    left = (width - 224) // 2
    top = (height - 224) // 2
    return image.crop((left, top, left + 224, top + 224))


def image_stats(task) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, str, str]:
    path_text, allow_variable_size, decode_transform = task
    path = Path(path_text)
    with Image.open(path) as image:
        image = image.convert("RGB")
        source_size = image.size
        image = apply_decode_transform(image, decode_transform)
        decoded_size = image.size
        if not allow_variable_size and decoded_size != (224, 224):
            raise RuntimeError(f"not 224x224: {path} ({image.size})")
        pixels = np.asarray(image, dtype=np.float64).reshape(-1, 3) / 255.0
    square = np.square(pixels)
    return (
        pixels.sum(axis=0), square.sum(axis=0), pixels.shape[0],
        pixels.mean(axis=0), square.mean(axis=0),
        f"{source_size[0]}x{source_size[1]}",
        f"{decoded_size[0]}x{decoded_size[1]}",
    )


def main() -> None:
    parser = argparse.ArgumentParser("Compute exact channel statistics for a prepared dataset split")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--allow-variable-size", action="store_true")
    parser.add_argument(
        "--decode-transform", default="none",
        choices=("none", "warp224", "shorter224_centercrop224", "shorter256_centercrop224"),
    )
    parser.add_argument("--include-stems-file", type=Path,
                        help="optional split list whose first whitespace-delimited field is an image stem")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cfg = get_dataset(args.dataset_name)
    split_dir = args.data_dir / args.split
    paths = sorted(
        path for path in split_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    requested_stems = None
    if args.include_stems_file:
        requested_stems = {
            line.split(maxsplit=1)[0]
            for line in args.include_stems_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        paths = [path for path in paths if path.stem in requested_stems]
        observed_stems = {path.stem for path in paths}
        if observed_stems != requested_stems:
            missing = sorted(requested_stems - observed_stems)
            raise RuntimeError(f"split list stems missing from ImageFolder: {missing[:10]}")
    if not paths:
        raise RuntimeError(f"no images found: {split_dir}")

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    equal_image_mean_sum = np.zeros(3, dtype=np.float64)
    equal_image_square_mean_sum = np.zeros(3, dtype=np.float64)
    pixels = 0
    source_sizes = Counter()
    decoded_sizes = Counter()
    with Pool(processes=args.workers) as pool:
        tasks = (
            (str(path), args.allow_variable_size, args.decode_transform)
            for path in paths
        )
        for (partial_sum, partial_square_sum, partial_pixels,
             image_mean, image_square_mean, source_size, decoded_size) in pool.imap_unordered(
            image_stats, tasks, chunksize=16
        ):
            channel_sum += partial_sum
            channel_square_sum += partial_square_sum
            equal_image_mean_sum += image_mean
            equal_image_square_mean_sum += image_square_mean
            pixels += partial_pixels
            source_sizes[source_size] += 1
            decoded_sizes[decoded_size] += 1
    mean = channel_sum / pixels
    variance = np.maximum(channel_square_sum / pixels - np.square(mean), 0.0)
    std = np.sqrt(variance)
    equal_image_mean = equal_image_mean_sum / len(paths)
    equal_image_variance = np.maximum(
        equal_image_square_mean_sum / len(paths) - np.square(equal_image_mean), 0.0
    )
    equal_image_std = np.sqrt(equal_image_variance)
    expected_mean = np.asarray(cfg.mean)
    expected_std = np.asarray(cfg.std)
    payload = {
        "status": "complete",
        "dataset": cfg.name,
        "split": args.split,
        "data_dir": str(args.data_dir.resolve()),
        "images": len(paths),
        "include_stems_file": (
            str(args.include_stems_file.resolve()) if args.include_stems_file else None
        ),
        "pixels": pixels,
        "decode_transform": args.decode_transform,
        "source_sizes": dict(sorted(source_sizes.items())),
        "decoded_sizes": dict(sorted(decoded_sizes.items())),
        "mean": mean.tolist(),
        "std_population": std.tolist(),
        "mean_equal_image_weight": equal_image_mean.tolist(),
        "std_equal_image_weight": equal_image_std.tolist(),
        "official_mean": list(cfg.mean),
        "official_std": list(cfg.std),
        "mean_delta": (mean - expected_mean).tolist(),
        "std_delta": (std - expected_std).tolist(),
        "max_abs_mean_delta": float(np.max(np.abs(mean - expected_mean))),
        "max_abs_std_delta": float(np.max(np.abs(std - expected_std))),
        "statistic_definition": "mean/std are pixel-weighted population statistics; equal_image fields weight each image equally; decoded RGB is scaled to [0,1]",
    }
    output = args.output or args.data_dir / f"{args.split}_channel_stats.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
