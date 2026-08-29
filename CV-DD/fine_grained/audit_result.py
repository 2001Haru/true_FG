import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Audit a fine-grained post-evaluation result")
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--classes", required=True, type=int)
    parser.add_argument("--validation-images", required=True, type=int)
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(f"result audit failed: {message}")


def main() -> None:
    args = parse_args()
    if not args.result.is_file():
        fail(f"missing result: {args.result}")
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    required = {
        "best_top1", "training_target", "num_classes", "validation_images",
        "primary_metric", "native_top1_at_best_checkpoint", "per_class",
    }
    missing = sorted(required - payload.keys())
    if missing:
        fail(f"missing keys: {missing}")
    if payload["training_target"] != "fkd_soft_label":
        fail(f"unexpected training target: {payload['training_target']}")
    if payload["primary_metric"] != "native_top1":
        fail(f"unexpected primary metric: {payload['primary_metric']}")
    if payload["num_classes"] != args.classes:
        fail(f"num_classes={payload['num_classes']} != {args.classes}")
    if payload["validation_images"] != args.validation_images:
        fail(f"validation_images={payload['validation_images']} != {args.validation_images}")

    best = float(payload["best_top1"])
    native = float(payload["native_top1_at_best_checkpoint"])
    if not (math.isfinite(best) and 0.0 <= best <= 100.0):
        fail(f"invalid best_top1: {best}")
    if not math.isclose(best, native, rel_tol=0.0, abs_tol=1e-4):
        fail(f"best_top1={best} differs from reloaded-checkpoint Top-1={native}")

    per_class = payload["per_class"]
    if not isinstance(per_class, list) or len(per_class) != args.classes:
        fail(f"per_class length is not {args.classes}")
    total_images = 0
    total_correct = 0
    for expected_id, row in enumerate(per_class):
        if row.get("class_id") != expected_id:
            fail(f"class_id mismatch at index {expected_id}: {row.get('class_id')}")
        total = int(row["total"])
        correct = int(row["correct"])
        accuracy = float(row["accuracy"])
        if total <= 0 or correct < 0 or correct > total:
            fail(f"invalid counts for class {expected_id}: {correct}/{total}")
        expected_accuracy = 100.0 * correct / total
        if not math.isclose(accuracy, expected_accuracy, rel_tol=0.0, abs_tol=1e-8):
            fail(f"incorrect per-class accuracy for class {expected_id}")
        total_images += total
        total_correct += correct
    if total_images != args.validation_images:
        fail(f"per-class totals sum to {total_images}, expected {args.validation_images}")
    reconstructed = 100.0 * total_correct / total_images
    if not math.isclose(native, reconstructed, rel_tol=0.0, abs_tol=1e-4):
        fail(f"native Top-1={native} differs from per-class reconstruction={reconstructed}")
    print(json.dumps({
        "status": "complete",
        "result": str(args.result.resolve()),
        "best_top1": best,
        "classes": args.classes,
        "validation_images": args.validation_images,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
