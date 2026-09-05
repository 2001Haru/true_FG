"""Audit radial mismatch between rival-facing and outward DINO selections."""

import argparse
import csv
import json
import os
import statistics
from pathlib import Path

import numpy as np


DATASETS = {
    "CUB_imsize224": 200,
    "A_imsize224": 100,
    "SC_imsize224": 196,
}


def quantiles(values) -> dict:
    array = np.asarray(values, dtype=float)
    points = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 1.0])
    return dict(zip(("min", "q25", "median", "q75", "max"), map(float, points)))


def describe(values) -> dict:
    values = list(map(float, values))
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        **quantiles(values),
    }


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + end - 1) / 2
        position = end
    return ranks


def class_accuracy(experiment_root: Path, dataset: str, arm: str, classes: int):
    values = {class_id: [] for class_id in range(classes)}
    totals = {}
    files = sorted((experiment_root / "results" / dataset / arm).glob("sseed*.json"))
    if len(files) != 6:
        raise RuntimeError(f"{dataset}/{arm}: expected six Student results, found {len(files)}")
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or payload.get("updates_completed") != 3000:
            raise RuntimeError(f"invalid result: {path}")
        for row in payload["per_class_final"]:
            class_id = int(row["class_id"])
            values[class_id].append(float(row["accuracy"]))
            totals[class_id] = int(row["total"])
    return {class_id: statistics.mean(rows) for class_id, rows in values.items()}, totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    all_rows = []
    summaries = {}
    for dataset, classes in DATASETS.items():
        manifest_root = args.experiment_root / "selections" / dataset / "manifests"
        rival_rows = json.loads(
            (manifest_root / "rival_facing_edge.json").read_text(encoding="utf-8")
        )["images"]
        outward_rows = json.loads(
            (manifest_root / "outward_edge.json").read_text(encoding="utf-8")
        )["images"]
        rival = {int(row["class_id"]): row for row in rival_rows}
        outward = {int(row["class_id"]): row for row in outward_rows}
        if sorted(rival) != list(range(classes)) or sorted(outward) != list(range(classes)):
            raise RuntimeError(f"selection manifests do not cover every {dataset} class")
        rival_accuracy, totals = class_accuracy(
            args.experiment_root, dataset, "rival_facing_edge", classes
        )
        outward_accuracy, outward_totals = class_accuracy(
            args.experiment_root, dataset, "outward_edge", classes
        )
        if totals != outward_totals:
            raise RuntimeError(f"validation class totals differ between arms for {dataset}")
        dataset_rows = []
        for class_id in range(classes):
            rival_row = rival[class_id]
            outward_row = outward[class_id]
            row = {
                "dataset": dataset,
                "class_id": class_id,
                "class_folder": rival_row["class_folder"],
                "rival_source_path": rival_row["source_path"],
                "outward_source_path": outward_row["source_path"],
                "rival_radial_percentile": float(
                    rival_row["within_class_radial_percentile"]
                ),
                "outward_radial_percentile": float(
                    outward_row["within_class_radial_percentile"]
                ),
                "delta_radial_percentile": float(
                    outward_row["within_class_radial_percentile"]
                    - rival_row["within_class_radial_percentile"]
                ),
                "rival_radial_cosine_distance": float(
                    rival_row["radial_cosine_distance"]
                ),
                "outward_radial_cosine_distance": float(
                    outward_row["radial_cosine_distance"]
                ),
                "delta_radial_cosine_distance": float(
                    outward_row["radial_cosine_distance"]
                    - rival_row["radial_cosine_distance"]
                ),
                "rival_nearest_similarity": float(
                    rival_row["nearest_rival_similarity"]
                ),
                "outward_nearest_similarity": float(
                    outward_row["nearest_rival_similarity"]
                ),
                "delta_nearest_rival_similarity": float(
                    outward_row["nearest_rival_similarity"]
                    - rival_row["nearest_rival_similarity"]
                ),
                "rival_final_per_class_accuracy_mean_6_students": rival_accuracy[class_id],
                "outward_final_per_class_accuracy_mean_6_students": outward_accuracy[class_id],
                "delta_final_per_class_accuracy": (
                    outward_accuracy[class_id] - rival_accuracy[class_id]
                ),
                "validation_images": totals[class_id],
            }
            dataset_rows.append(row)
            all_rows.append(row)
        delta_percentile = np.asarray(
            [row["delta_radial_percentile"] for row in dataset_rows]
        )
        delta_accuracy = np.asarray(
            [row["delta_final_per_class_accuracy"] for row in dataset_rows]
        )
        weights = np.asarray([row["validation_images"] for row in dataset_rows], dtype=float)
        matched = {}
        for threshold in (0.025, 0.05, 0.10, 0.15):
            mask = np.abs(delta_percentile) <= threshold + 1e-12
            matched[str(threshold)] = {
                "classes": int(mask.sum()),
                "unweighted_delta_accuracy": (
                    float(delta_accuracy[mask].mean()) if mask.any() else None
                ),
                "test_count_weighted_delta_accuracy": (
                    float(np.average(delta_accuracy[mask], weights=weights[mask]))
                    if mask.any()
                    else None
                ),
                "wins": int((delta_accuracy[mask] > 1e-12).sum()),
                "ties": int((np.abs(delta_accuracy[mask]) <= 1e-12).sum()),
                "losses": int((delta_accuracy[mask] < -1e-12).sum()),
            }
        summaries[dataset] = {
            "classes": classes,
            "rival_radial_percentile": describe(
                row["rival_radial_percentile"] for row in dataset_rows
            ),
            "outward_radial_percentile": describe(
                row["outward_radial_percentile"] for row in dataset_rows
            ),
            "delta_radial_percentile": describe(delta_percentile),
            "absolute_delta_radial_percentile": describe(np.abs(delta_percentile)),
            "delta_radial_cosine_distance": describe(
                row["delta_radial_cosine_distance"] for row in dataset_rows
            ),
            "delta_nearest_rival_similarity": describe(
                row["delta_nearest_rival_similarity"] for row in dataset_rows
            ),
            "outward_higher_equal_lower_percentile": {
                "higher": int((delta_percentile > 1e-12).sum()),
                "equal": int((np.abs(delta_percentile) <= 1e-12).sum()),
                "lower": int((delta_percentile < -1e-12).sum()),
            },
            "pearson_delta_percentile_vs_delta_accuracy": float(
                np.corrcoef(delta_percentile, delta_accuracy)[0, 1]
            ),
            "spearman_delta_percentile_vs_delta_accuracy": float(
                np.corrcoef(
                    average_ranks(delta_percentile), average_ranks(delta_accuracy)
                )[0, 1]
            ),
            "approximately_radius_matched_subsets": matched,
            "full_test_count_weighted_outward_minus_rival_accuracy": float(
                np.average(delta_accuracy, weights=weights)
            ),
        }
    payload = {
        "status": "complete",
        "experiment": "dino_sixarm_ipc1",
        "comparison": "outward_edge minus rival_facing_edge",
        "finding": (
            "shared percentile support does not match realized radii; causal direction-only "
            "interpretation is not identified by the existing selections"
        ),
        "summaries": summaries,
        "class_rows": all_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_json)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps({"status": "complete", "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()

