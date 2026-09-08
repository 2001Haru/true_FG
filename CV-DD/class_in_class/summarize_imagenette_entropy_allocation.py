"""Summarize paired within-manifest Soft-1 allocation and entropy Top10 results."""

import argparse
import json
import os
import statistics
from pathlib import Path


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stats(values):
    values = list(map(float, values))
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "values": values,
        "positive": sum(value > 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "negative": sum(value < 0 for value in values),
    }


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--experiment-name", default="imagenette_entropy_allocation_v1")
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    conditions = {name: [] for name in ("all_hard", "high_entropy_soft1", "low_entropy_soft1", "all_soft1")}
    contrasts = {name: [] for name in ("high_minus_low", "high_minus_hard", "low_minus_hard", "all_soft_minus_hard", "allocation_nonadditivity")}
    tuples = []
    errors = []
    for rseed in (0, 1, 2):
        for sseed in (42, 43, 44):
            paths = {
                "all_hard": root / "results" / "lambda_+0" / f"rseed{rseed}" / f"hard_sseed{sseed}.json",
                "high_entropy_soft1": root / "results" / "lambda0_entropy_allocation" / f"rseed{rseed}" / f"high_entropy_soft1_sseed{sseed}.json",
                "low_entropy_soft1": root / "results" / "lambda0_entropy_allocation" / f"rseed{rseed}" / f"low_entropy_soft1_sseed{sseed}.json",
                "all_soft1": root / "results" / "lambda_+0" / f"rseed{rseed}" / f"soft1_sseed{sseed}.json",
            }
            if any(not path.is_file() for path in paths.values()):
                errors.append(f"missing lambda0 allocation tuple r{rseed}/s{sseed}")
                continue
            rows = {name: load(path) for name, path in paths.items()}
            manifest_hashes = {row["train_manifest_sha256"] for row in rows.values()}
            initial_hashes = {row["initial_student_state_sha256"] for row in rows.values()}
            if len(manifest_hashes) != 1 or len(initial_hashes) != 1:
                errors.append(f"pairing mismatch r{rseed}/s{sseed}")
            accuracy = {name: float(row["final_top1"]) for name, row in rows.items()}
            for name, value in accuracy.items():
                conditions[name].append(value)
            values = {
                "high_minus_low": accuracy["high_entropy_soft1"] - accuracy["low_entropy_soft1"],
                "high_minus_hard": accuracy["high_entropy_soft1"] - accuracy["all_hard"],
                "low_minus_hard": accuracy["low_entropy_soft1"] - accuracy["all_hard"],
                "all_soft_minus_hard": accuracy["all_soft1"] - accuracy["all_hard"],
                "allocation_nonadditivity": accuracy["all_soft1"] + accuracy["all_hard"] - accuracy["high_entropy_soft1"] - accuracy["low_entropy_soft1"],
            }
            for name, value in values.items():
                contrasts[name].append(value)
            tuples.append({"selection_seed": rseed, "student_seed": sseed, **accuracy, **values})

    top = {"hard": [], "soft1": []}
    top_gain = []
    top_runs = []
    for sseed in (42, 43, 44):
        hard_path = root / "results" / "highest_entropy_top10" / f"hard_sseed{sseed}.json"
        soft_path = root / "results" / "highest_entropy_top10" / f"soft1_sseed{sseed}.json"
        if not hard_path.is_file() or not soft_path.is_file():
            errors.append(f"missing Top10 pair s{sseed}")
            continue
        hard, soft = load(hard_path), load(soft_path)
        if hard["train_manifest_sha256"] != soft["train_manifest_sha256"] or hard["initial_student_state_sha256"] != soft["initial_student_state_sha256"]:
            errors.append(f"Top10 pairing mismatch s{sseed}")
        h, s = float(hard["final_top1"]), float(soft["final_top1"])
        top["hard"].append(h); top["soft1"].append(s); top_gain.append(s - h)
        top_runs.append({"student_seed": sseed, "hard": h, "soft1": s, "soft1_minus_hard": s - h})

    expected = 24
    found = sum(len(values) for values in (conditions["high_entropy_soft1"], conditions["low_entropy_soft1"], top["hard"], top["soft1"]))
    payload = {
        "status": "complete" if found == expected and not errors else "incomplete",
        "experiment": args.experiment_name,
        "expected_new_results": expected,
        "found_new_results": found,
        "errors": errors,
        "lambda0_conditions": {name: stats(values) for name, values in conditions.items()},
        "lambda0_paired_contrasts": {name: stats(values) for name, values in contrasts.items()},
        "lambda0_tuples": tuples,
        "highest_entropy_top10": {
            "hard": stats(top["hard"]),
            "soft1": stats(top["soft1"]),
            "paired_soft1_minus_hard": stats(top_gain),
            "runs": top_runs,
        },
        "primary_metric": "Final Top-1 at update4000",
    }
    atomic_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("status", "expected_new_results", "found_new_results", "errors")}, indent=2))
    if payload["status"] != "complete":
        raise RuntimeError("allocation matrix incomplete")


if __name__ == "__main__":
    main()
