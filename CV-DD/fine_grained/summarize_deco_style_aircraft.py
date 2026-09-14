"""Summarize controlled DeCO-style Aircraft IPC3 results."""
import argparse
import json
import statistics
from pathlib import Path


def stats(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "values": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--r0-results", required=True, type=Path)
    parser.add_argument("--construction-manifest", type=Path)
    parser.add_argument("--protocol", default="controlled_deco_style_regions_v1")
    args = parser.parse_args()
    roots = {
        "r0": args.r0_results,
        "random_regions": args.root / "results/random_regions",
        "fg_regions": args.root / "results/fg_regions",
    }
    rows = {name: [json.loads((root / f"ipc3_sseed{seed}.json").read_text()) for seed in (42, 43, 44)] for name, root in roots.items()}
    metrics = ("best_top1", "final_epoch_top1")
    construction_manifest = args.construction_manifest or args.root / "construction/construction_manifest.json"
    construction = json.loads(construction_manifest.read_text())
    result = {
        "status": "complete", "protocol": args.protocol,
        "construction_manifest": str(construction_manifest.resolve()),
        "groups": {name: {metric: stats([row[metric] for row in group]) for metric in metrics} for name, group in rows.items()},
        "paired_deltas": {},
        "storage": {key: construction[key] for key in ("ipc", "stored_images", "regions", "regions_per_class", "independent_sources_per_class", "window_area_ratio")},
    }
    for candidate, reference in (("random_regions", "r0"), ("fg_regions", "r0"), ("fg_regions", "random_regions")):
        result["paired_deltas"][f"{candidate} minus {reference}"] = {
            metric: stats([left[metric] - right[metric] for left, right in zip(rows[candidate], rows[reference])])
            for metric in metrics
        }
    for method in ("random_regions", "fg_regions"):
        for candidate, reference in zip(rows[method], rows["r0"]):
            assert candidate["student_seed"] == reference["student_seed"]
            if "initial_model_sha256" in reference:
                assert candidate["initial_model_sha256"] == reference["initial_model_sha256"]
            assert candidate["temperature"] == 20 and candidate["epochs"] == 400
    output = args.root / "summary/summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
