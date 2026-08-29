import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser("Compare two paired student result JSON files")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    for key in ("num_classes", "validation_images", "primary_metric"):
        if baseline.get(key) != candidate.get(key):
            raise RuntimeError(f"paired result mismatch for {key}")
    baseline_rows = baseline["per_class"]
    candidate_rows = candidate["per_class"]
    if len(baseline_rows) != len(candidate_rows):
        raise RuntimeError("per-class lengths differ")

    rows = []
    for left, right in zip(baseline_rows, candidate_rows):
        for key in ("class_id", "class_name", "total"):
            if left[key] != right[key]:
                raise RuntimeError(f"per-class mismatch for {key}: {left} / {right}")
        rows.append({
            "class_id": left["class_id"],
            "class_name": left["class_name"],
            "total": left["total"],
            "baseline_accuracy": float(left["accuracy"]),
            "candidate_accuracy": float(right["accuracy"]),
            "delta": float(right["accuracy"]) - float(left["accuracy"]),
        })
    deltas = [row["delta"] for row in rows]
    payload = {
        "status": "complete",
        "baseline": {
            "label": args.baseline_label,
            "path": str(args.baseline.resolve()),
            "top1": float(baseline["best_top1"]),
        },
        "candidate": {
            "label": args.candidate_label,
            "path": str(args.candidate.resolve()),
            "top1": float(candidate["best_top1"]),
        },
        "top1_delta": float(candidate["best_top1"]) - float(baseline["best_top1"]),
        "per_class": {
            "mean_delta": statistics.mean(deltas),
            "median_delta": statistics.median(deltas),
            "improved_classes": sum(delta > 0 for delta in deltas),
            "equal_classes": sum(delta == 0 for delta in deltas),
            "worse_classes": sum(delta < 0 for delta in deltas),
            "largest_improvements": sorted(rows, key=lambda row: row["delta"], reverse=True)[:10],
            "largest_regressions": sorted(rows, key=lambda row: row["delta"])[:10],
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
