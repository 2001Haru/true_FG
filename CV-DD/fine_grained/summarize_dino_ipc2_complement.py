"""Audit and summarize the 180-result DINO IPC2 complement matrix."""

import argparse
import json
import math
import os
import statistics
from pathlib import Path


DATASETS = {
    "CUB_imsize224": (200, 5794),
    "A_imsize224": (100, 3333),
    "SC_imsize224": (196, 8041),
}
STOCHASTIC = (
    "random_ipc2",
    "spherical_kmeans2",
    "center_plus_random",
    "center_plus_shell_random",
)
DETERMINISTIC = (
    "center_plus_outward",
    "center_plus_high_margin",
    "center_plus_rival_facing",
    "global_center_top2",
)


def same(actual, expected) -> bool:
    if isinstance(expected, float):
        return actual is not None and math.isclose(float(actual), expected, abs_tol=1e-12)
    return actual == expected


def aggregate(rows: list[dict]) -> dict:
    final = [row["final_top1"] for row in rows]
    best = [row["best_top1_diagnostic"] for row in rows]
    return {
        "runs": rows,
        "primary_final_mean": statistics.mean(final),
        "primary_final_sample_std": statistics.stdev(final),
        "diagnostic_best_mean": statistics.mean(best),
        "diagnostic_best_sample_std": statistics.stdev(best),
    }


def validate_result(
    path: Path,
    payload: dict,
    dataset: str,
    classes: int,
    validation_images: int,
    method: str,
    arm: str,
    selection_seed,
    student_seed: int,
) -> list[str]:
    expected = {
        "status": "complete",
        "protocol": "hard_label_v1",
        "method": "DINOIPC2Selection",
        "dataset": dataset,
        "ipc": 2,
        "selection_experiment": "dino_ipc2_center_complement",
        "selection_method": method,
        "selection_arm": arm,
        "selection_seed": selection_seed,
        "student_seed": student_seed,
        "student_initialization": "imagenet1k_v1",
        "total_optimizer_updates": 3000,
        "updates_completed": 3000,
        "physical_batch_size": 64,
        "gradient_accumulation_steps": 1,
        "num_classes": classes,
        "train_images": 2 * classes,
        "validation_images": validation_images,
        "primary_metric": "final_update_top1",
        "geometry_recomputed": False,
        "training_sample_weighting": "equal; both images mixed by the same shuffled loader",
        "train_source_type": "selection_manifest",
        "eval_openblas_num_threads": 1,
    }
    return [
        f"{path}: {key}={payload.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if not same(payload.get(key), value)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--parent-ipc1-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    errors = []
    found = 0
    summary = {}
    parent = json.loads(args.parent_ipc1_summary.read_text(encoding="utf-8"))
    if parent.get("status") != "complete" or parent.get("found_results") != 126:
        errors.append("parent IPC1 six-arm summary is incomplete")
    for dataset, (classes, validation_images) in DATASETS.items():
        audit_path = args.experiment_root / "selection_audits" / f"{dataset}.json"
        if not audit_path.is_file():
            errors.append(f"missing selection audit: {audit_path}")
            selection_audit = {}
        else:
            selection_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if (
                selection_audit.get("status") != "complete"
                or selection_audit.get("classes") != classes
                or selection_audit.get("center_collisions") != []
                or selection_audit.get("duplicate_pairs") != []
            ):
                errors.append(f"invalid selection audit: {audit_path}")
        dataset_summary = {"selection_audit": str(audit_path.resolve())}
        parent_centroid = {
            row["student_seed"]: row["final_top1"]
            for row in parent["summary"][dataset]["centroid"]["runs"]
        }
        for method in STOCHASTIC:
            rows = []
            for selection_seed in (0, 1, 2):
                arm = f"{method}_rseed{selection_seed}"
                for student_seed in (42, 43, 44):
                    path = (
                        args.experiment_root / "results" / dataset / arm /
                        f"sseed{student_seed}.json"
                    )
                    if not path.is_file():
                        errors.append(f"missing result: {path}")
                        continue
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    found += 1
                    errors.extend(
                        validate_result(
                            path, payload, dataset, classes, validation_images,
                            method, arm, selection_seed, student_seed,
                        )
                    )
                    rows.append(
                        {
                            "selection_seed": selection_seed,
                            "student_seed": student_seed,
                            "final_top1": float(payload["final_top1"]),
                            "best_top1_diagnostic": float(payload["best_top1_diagnostic"]),
                            "best_update_diagnostic": int(payload["best_update_diagnostic"]),
                            "path": str(path.resolve()),
                        }
                    )
            if len(rows) == 9:
                result = aggregate(rows)
                result["final_mean_by_selection_seed"] = {
                    str(seed): statistics.mean(
                        row["final_top1"] for row in rows if row["selection_seed"] == seed
                    )
                    for seed in (0, 1, 2)
                }
                if method.startswith("center_plus_"):
                    deltas = [
                        row["final_top1"] - parent_centroid[row["student_seed"]]
                        for row in rows
                    ]
                    result["delta_vs_ipc1_centroid_index_aligned"] = {
                        "mean": statistics.mean(deltas),
                        "sample_std": statistics.stdev(deltas),
                        "wins": sum(value > 0 for value in deltas),
                        "ties": sum(value == 0 for value in deltas),
                        "losses": sum(value < 0 for value in deltas),
                        "values": deltas,
                    }
                dataset_summary[method] = result
        for method in DETERMINISTIC:
            rows = []
            arm = method
            for student_seed in range(42, 48):
                path = (
                    args.experiment_root / "results" / dataset / arm /
                    f"sseed{student_seed}.json"
                )
                if not path.is_file():
                    errors.append(f"missing result: {path}")
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                found += 1
                errors.extend(
                    validate_result(
                        path, payload, dataset, classes, validation_images,
                        method, arm, None, student_seed,
                    )
                )
                rows.append(
                    {
                        "student_seed": student_seed,
                        "final_top1": float(payload["final_top1"]),
                        "best_top1_diagnostic": float(payload["best_top1_diagnostic"]),
                        "best_update_diagnostic": int(payload["best_update_diagnostic"]),
                        "path": str(path.resolve()),
                    }
                )
            if len(rows) == 6:
                result = aggregate(rows)
                deltas = [
                    row["final_top1"] - parent_centroid[row["student_seed"]]
                    for row in rows
                ]
                result["paired_delta_vs_ipc1_centroid"] = {
                    "mean": statistics.mean(deltas),
                    "sample_std": statistics.stdev(deltas),
                    "wins": sum(value > 0 for value in deltas),
                    "ties": sum(value == 0 for value in deltas),
                    "losses": sum(value < 0 for value in deltas),
                    "values": deltas,
                }
                dataset_summary[method] = result
        summary[dataset] = dataset_summary
    payload = {
        "status": "complete" if found == 180 and not errors else "incomplete",
        "experiment": "dino_ipc2_center_complement",
        "protocol": "hard_label_v1",
        "primary_metric": "final_update_top1",
        "expected_results": 180,
        "found_results": found,
        "stochastic_selection_seeds": [0, 1, 2],
        "stochastic_student_seeds": [42, 43, 44],
        "deterministic_student_seeds": [42, 43, 44, 45, 46, 47],
        "parent_ipc1_summary": str(args.parent_ipc1_summary.resolve()),
        "errors": errors,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "complete":
        raise RuntimeError(f"IPC2 matrix incomplete: {len(errors)} errors")


if __name__ == "__main__":
    main()
