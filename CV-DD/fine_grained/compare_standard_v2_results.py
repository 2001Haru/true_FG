"""Paired comparisons for the completed fine-grained standard-v2 matrix."""

import argparse
import json
import math
import os
import statistics
from pathlib import Path

from scipy.stats import t


DATASETS = ("CUB_imsize224", "A_imsize224", "SC_imsize224")
SEEDS = (42, 43, 44)
IPCS = (1, 3, 5)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values):
    values = list(map(float, values))
    mean = statistics.mean(values)
    sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
    se = sample_std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    critical = float(t.ppf(0.975, len(values) - 1)) if len(values) > 1 else 0.0
    return {
        "count": len(values), "mean": mean, "sample_std": sample_std,
        "ci95_paired_t": [mean - critical * se, mean + critical * se],
        "positive": sum(value > 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "negative": sum(value < 0 for value in values), "values": values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", required=True, type=Path)
    parser.add_argument("--v1-root", required=True, type=Path)
    parser.add_argument("--cub4k-root", required=True, type=Path)
    parser.add_argument("--legacy-cub-imagenet-root", required=True, type=Path)
    args = parser.parse_args()
    paired_random, paired_legacy = [], []
    for dataset in DATASETS:
        for ipc in IPCS:
            for teacher_seed in SEEDS:
                for recovery_seed in SEEDS:
                    for student_seed in SEEDS:
                        relative = (Path("results") / f"tseed{teacher_seed}" / dataset /
                                    f"rseed{recovery_seed}" / f"ipc{ipc}_sseed{student_seed}.json")
                        v2 = load(args.v2_root / relative)
                        baseline_root = args.cub4k_root if dataset == "CUB_imsize224" else args.v1_root
                        baseline = load(baseline_root / relative)
                        paired_random.append({
                            "dataset": dataset, "ipc": ipc, "teacher_seed": teacher_seed,
                            "recovery_seed": recovery_seed, "student_seed": student_seed,
                            "best_delta_v2_minus_random_v1": v2["best_top1"] - baseline["best_top1"],
                            "final_delta_v2_minus_random_v1": v2["final_epoch_top1"] - baseline["final_epoch_top1"],
                        })
            for student_seed in SEEDS:
                v2 = load(args.v2_root / "results/tseed42/CUB_imsize224/rseed42" /
                          f"ipc{ipc}_sseed{student_seed}.json") if dataset == "CUB_imsize224" else None
                if v2 is None:
                    continue
                legacy = load(args.legacy_cub_imagenet_root / "results/iter4000/CUB_imsize224/rseed42" /
                              f"ipc{ipc}_sseed{student_seed}.json")
                paired_legacy.append({
                    "dataset": dataset, "ipc": ipc, "teacher_seed": 42,
                    "recovery_seed": 42, "student_seed": student_seed,
                    "best_delta_v2_minus_legacy_imagenet": v2["best_top1"] - legacy["best_top1"],
                    "final_delta_v2_minus_legacy_imagenet": v2["final_epoch_top1"] - legacy["final_epoch_top1"],
                })
    groups_random, groups_legacy = [], []
    for dataset in DATASETS:
        for ipc in IPCS:
            selected = [row for row in paired_random if row["dataset"] == dataset and row["ipc"] == ipc]
            groups_random.append({
                "dataset": dataset, "ipc": ipc,
                "best_delta_v2_minus_random_v1": stats(row["best_delta_v2_minus_random_v1"] for row in selected),
                "final_delta_v2_minus_random_v1": stats(row["final_delta_v2_minus_random_v1"] for row in selected),
            })
    for ipc in IPCS:
        selected = [row for row in paired_legacy if row["ipc"] == ipc]
        groups_legacy.append({
            "dataset": "CUB_imsize224", "ipc": ipc,
            "best_delta_v2_minus_legacy_imagenet": stats(row["best_delta_v2_minus_legacy_imagenet"] for row in selected),
            "final_delta_v2_minus_legacy_imagenet": stats(row["final_delta_v2_minus_legacy_imagenet"] for row in selected),
        })
    payload = {
        "status": "complete" if len(paired_random) == 243 and len(paired_legacy) == 9 else "failed",
        "comparison_to_random_v1": {
            "changes": "student initialization plus differential LR plus full cosine schedule",
            "paired_runs": len(paired_random), "groups": groups_random, "rows": paired_random,
        },
        "comparison_to_legacy_cub_imagenet": {
            "changes": "differential LR plus full cosine schedule; same ImageNet initialization and same t42/r42 4k upstream",
            "paired_runs": len(paired_legacy), "groups": groups_legacy, "rows": paired_legacy,
        },
    }
    output = args.v2_root / "summary/v2_paired_comparisons.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": payload["status"], "random_pairs": len(paired_random),
                      "legacy_imagenet_pairs": len(paired_legacy), "output": str(output.resolve())}))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
