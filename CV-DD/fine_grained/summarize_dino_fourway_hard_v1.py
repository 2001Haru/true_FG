"""Summarize the IPC1 DINO geometry four-way hard-label v1 experiment."""

import argparse
import hashlib
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
DETERMINISTIC_ARMS = ("centroid", "inter_class_boundary", "outward_frontier")
RANDOM_ARMS = ("random_rseed0", "random_rseed1", "random_rseed2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    errors = []
    found = 0
    summary = {}
    for dataset, (classes, validation_images) in DATASETS.items():
        dataset_summary = {}
        audit_path = args.experiment_root / "selection_audits" / f"{dataset}.json"
        if not audit_path.is_file():
            errors.append(f"missing geometry audit: {audit_path}")
            audit_hash = None
        else:
            audit_hash = sha256(audit_path)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if audit.get("status") != "complete" or audit.get("classes") != classes:
                errors.append(f"invalid geometry audit: {audit_path}")
            class_summaries = audit.get("class_summaries", [])
            if len(class_summaries) != classes:
                errors.append(f"geometry class summaries {len(class_summaries)} != {classes}")
            else:
                dataset_summary["selection_geometry"] = {
                    "outward_min_rival_overlap_rate": audit.get("outward_min_rival_overlap_rate"),
                    "shell_size_min": min(row["shell_images"] for row in class_summaries),
                    "shell_size_max": max(row["shell_images"] for row in class_summaries),
                    "shell_size_mean": statistics.mean(
                        row["shell_images"] for row in class_summaries
                    ),
                }
        for method in DETERMINISTIC_ARMS:
            rows = []
            for student_seed in range(42, 48):
                result = args.experiment_root / "results" / dataset / method / f"sseed{student_seed}.json"
                if not result.is_file():
                    errors.append(f"missing result: {result}")
                    continue
                payload = json.loads(result.read_text(encoding="utf-8"))
                found += 1
                expected = {
                    "status": "complete",
                    "protocol": "hard_label_v1",
                    "method": "DINORealSelection",
                    "dataset": dataset,
                    "ipc": 1,
                    "selection_experiment": "dino_fourway_ipc1",
                    "selection_method": method,
                    "selection_arm": method,
                    "selection_seed": None,
                    "student_seed": student_seed,
                    "student_initialization": "imagenet1k_v1",
                    "total_optimizer_updates": 3000,
                    "updates_completed": 3000,
                    "physical_batch_size": 64,
                    "gradient_accumulation_steps": 1,
                    "num_classes": classes,
                    "validation_images": validation_images,
                    "primary_metric": "final_update_top1",
                    "eval_openblas_num_threads": 1,
                    "selection_audit_sha256": audit_hash,
                }
                for key, value in expected.items():
                    if not same(payload.get(key), value):
                        errors.append(f"{result}: {key} mismatch")
                manifest = Path(payload.get("selection_manifest", ""))
                if not manifest.is_file() or sha256(manifest) != payload.get("selection_manifest_sha256"):
                    errors.append(f"{result}: manifest missing or changed")
                rows.append(
                    {
                        "student_seed": student_seed,
                        "final_top1": float(payload["final_top1"]),
                        "best_top1_diagnostic": float(payload["best_top1_diagnostic"]),
                        "path": str(result.resolve()),
                    }
                )
            if len(rows) == 6:
                dataset_summary[method] = aggregate(rows)
        random_rows = []
        for selection_seed, arm in enumerate(RANDOM_ARMS):
            for student_seed in (42, 43, 44):
                result = args.experiment_root / "results" / dataset / arm / f"sseed{student_seed}.json"
                if not result.is_file():
                    errors.append(f"missing result: {result}")
                    continue
                payload = json.loads(result.read_text(encoding="utf-8"))
                found += 1
                expected = {
                    "status": "complete",
                    "protocol": "hard_label_v1",
                    "method": "DINORealSelection",
                    "dataset": dataset,
                    "ipc": 1,
                    "selection_experiment": "dino_fourway_ipc1",
                    "selection_method": "random",
                    "selection_arm": arm,
                    "selection_seed": selection_seed,
                    "student_seed": student_seed,
                    "student_initialization": "imagenet1k_v1",
                    "total_optimizer_updates": 3000,
                    "updates_completed": 3000,
                    "physical_batch_size": 64,
                    "gradient_accumulation_steps": 1,
                    "num_classes": classes,
                    "validation_images": validation_images,
                    "primary_metric": "final_update_top1",
                    "eval_openblas_num_threads": 1,
                    "selection_audit_sha256": audit_hash,
                }
                for key, value in expected.items():
                    if not same(payload.get(key), value):
                        errors.append(f"{result}: {key} mismatch")
                manifest = Path(payload.get("selection_manifest", ""))
                if not manifest.is_file() or sha256(manifest) != payload.get("selection_manifest_sha256"):
                    errors.append(f"{result}: manifest missing or changed")
                random_rows.append(
                    {
                        "selection_seed": selection_seed,
                        "student_seed": student_seed,
                        "final_top1": float(payload["final_top1"]),
                        "best_top1_diagnostic": float(payload["best_top1_diagnostic"]),
                        "path": str(result.resolve()),
                    }
                )
        if len(random_rows) == 9:
            random_summary = aggregate(random_rows)
            random_summary["final_mean_by_selection_seed"] = {
                str(seed): statistics.mean(
                    row["final_top1"] for row in random_rows if row["selection_seed"] == seed
                )
                for seed in (0, 1, 2)
            }
            dataset_summary["random"] = random_summary
        summary[dataset] = dataset_summary
    payload = {
        "status": "complete" if found == 81 and not errors else "incomplete",
        "protocol": "hard_label_v1",
        "experiment": "dino_fourway_ipc1",
        "primary_metric": "final_update_top1",
        "expected_results": 81,
        "found_results": found,
        "deterministic_student_seeds": [42, 43, 44, 45, 46, 47],
        "random_selection_seeds": [0, 1, 2],
        "random_student_seeds": [42, 43, 44],
        "errors": errors,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "complete":
        raise RuntimeError(f"four-way matrix incomplete: {len(errors)} errors")


if __name__ == "__main__":
    main()
