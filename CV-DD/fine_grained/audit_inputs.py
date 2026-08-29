import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from config import get_dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def inspect_split(path: Path, expected_classes: int, verify_all_sizes: bool) -> dict:
    classes = sorted(entry for entry in path.iterdir() if entry.is_dir())
    if len(classes) != expected_classes:
        raise RuntimeError(f"{path}: found {len(classes)} classes, expected {expected_classes}")
    counts = {}
    sizes = Counter()
    for class_dir in classes:
        images = sorted(
            entry for entry in class_dir.iterdir()
            if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
        )
        counts[class_dir.name] = len(images)
        inspected = images if verify_all_sizes else images[:1]
        for image_path in inspected:
            with Image.open(image_path) as image:
                sizes[f"{image.width}x{image.height}"] += 1
    return {
        "classes": [entry.name for entry in classes],
        "images": sum(counts.values()),
        "minimum_per_class": min(counts.values()),
        "maximum_per_class": max(counts.values()),
        "counts": counts,
        "inspected_sizes": dict(sizes),
    }


def main() -> None:
    parser = argparse.ArgumentParser("Audit physical fine-grained ImageFolder inputs")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-all-sizes", action="store_true")
    args = parser.parse_args()
    cfg = get_dataset(args.dataset_name)
    train = inspect_split(args.data_dir / "train", cfg.classes, args.verify_all_sizes)
    test = inspect_split(args.data_dir / "test", cfg.classes, args.verify_all_sizes)
    if train["classes"] != test["classes"]:
        raise RuntimeError("Train/test class directory ordering differs")
    if set(train["inspected_sizes"]) != {"224x224"} or set(test["inspected_sizes"]) != {"224x224"}:
        train_common = Counter(train["inspected_sizes"]).most_common(10)
        test_common = Counter(test["inspected_sizes"]).most_common(10)
        raise RuntimeError(
            "Physical image size mismatch: "
            f"train_unique={len(train['inspected_sizes'])} train_top10={train_common}; "
            f"test_unique={len(test['inspected_sizes'])} test_top10={test_common}"
        )
    payload = {
        "status": "valid",
        "dataset": cfg.to_dict(),
        "data_dir": str(args.data_dir.resolve()),
        "verify_all_sizes": args.verify_all_sizes,
        "train": train,
        "test": test,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
