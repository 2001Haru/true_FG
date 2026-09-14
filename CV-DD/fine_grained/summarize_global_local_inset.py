"""Summarize paired R0/global-local inset Student results."""
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
    args = parser.parse_args()
    construction = json.loads((args.root / "construction/construction_manifest.json").read_text())
    roots = {
        "r0": args.r0_results,
        "lowres_detail": args.root / "results/lowres_detail",
        "native_detail": args.root / "results/native_detail",
    }
    rows = {
        method: [json.loads((root / f"ipc3_sseed{seed}.json").read_text()) for seed in (42, 43, 44)]
        for method, root in roots.items()
    }
    metrics = ("best_top1", "final_epoch_top1")
    result = {
        "status": "complete",
        "protocol": "global_fullframe_plus_native_detail_v1",
        "construction_manifest": str((args.root / "construction/construction_manifest.json").resolve()),
        "modified_classes": construction["modified_classes"],
        "groups": {
            method: {metric: stats([row[metric] for row in group]) for metric in metrics}
            for method, group in rows.items()
        },
        "paired_deltas": {},
    }
    for candidate, reference in (("lowres_detail", "r0"), ("native_detail", "r0"), ("native_detail", "lowres_detail")):
        result["paired_deltas"][f"{candidate} minus {reference}"] = {
            metric: stats([left[metric] - right[metric] for left, right in zip(rows[candidate], rows[reference])])
            for metric in metrics
        }
    for method in ("lowres_detail", "native_detail"):
        for candidate, reference in zip(rows[method], rows["r0"]):
            assert candidate["initial_model_sha256"] == reference["initial_model_sha256"]
            assert candidate["student_seed"] == reference["student_seed"]
            assert candidate["temperature"] == 20 and candidate["epochs"] == 400
    output = args.root / "summary/summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
