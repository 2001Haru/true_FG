"""Audit lambda=0 high/low entropy halves and create the entropy Top10 manifest."""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from imagenette_entropy_protocol import CLASSES, IPC, atomic_json, file_sha256, validate_official_split


def distribution(rows, key):
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def group_metrics(rows):
    return {
        "images": len(rows),
        "calibration_entropy": distribution(rows, "calibration_entropy"),
        "holdout_entropy": distribution(rows, "holdout_entropy"),
        "calibration_true_probability_mean": statistics.mean(
            row["calibration_true_probability"] for row in rows
        ),
        "holdout_true_probability_mean": statistics.mean(
            row["holdout_true_probability"] for row in rows
        ),
        "calibration_teacher_error_rate": 1.0
        - statistics.mean(row["calibration_argmax_correct"] for row in rows),
        "holdout_teacher_error_rate": 1.0
        - statistics.mean(row["holdout_argmax_correct"] for row in rows),
        "dino_own_centroid_similarity_mean": statistics.mean(
            row["dino_own_centroid_similarity"] for row in rows
        ),
        "dino_radial_percentile_mean": statistics.mean(
            row["dino_radial_percentile"] for row in rows
        ),
    }


def split_manifest_rows(rows):
    low, high = [], []
    per_class = []
    for class_id in range(CLASSES):
        class_rows = [row for row in rows if int(row["class_id"]) == class_id]
        ordered = sorted(
            class_rows,
            key=lambda row: (float(row["calibration_entropy"]), row["relative_path"]),
        )
        if len(ordered) != IPC:
            raise RuntimeError(f"class {class_id} is not IPC10")
        class_low, class_high = ordered[: IPC // 2], ordered[IPC // 2 :]
        low.extend(class_low)
        high.extend(class_high)
        per_class.append(
            {
                "class_id": class_id,
                "class_name": ordered[0]["class_name"],
                "low": group_metrics(class_low),
                "high": group_metrics(class_high),
            }
        )
    return low, high, per_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    output = root / "preflight" / "lambda0_entropy_allocation_audit.json"
    if output.exists() and not args.force:
        raise RuntimeError(f"refusing overwrite: {output}")

    train_dataset, _ = validate_official_split(args.data_root.resolve())
    base_path = root / "preflight" / "selection_preflight.json"
    per_image_path = root / "preflight" / "per_image_teacher_entropy_and_dino.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    per_image = json.loads(per_image_path.read_text(encoding="utf-8"))
    rows = per_image["images"]
    expected_paths = [
        Path(path).relative_to(args.data_root.resolve()).as_posix()
        for path, _ in train_dataset.samples
    ]
    if [row["relative_path"] for row in rows] != expected_paths:
        raise RuntimeError("per-image statistics differ from official train split")
    manifests = {}
    seed_audits = {}
    for selection_seed in (0, 1, 2):
        manifest_path = root / "manifests" / f"lambda_{0:+d}_rseed{selection_seed}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_rows = manifest["images"]
        low, high, per_class = split_manifest_rows(manifest_rows)
        if len(low) != 50 or len(high) != 50:
            raise RuntimeError("lambda0 split is not exactly 50 low / 50 high")
        hard_result = root / "results" / "lambda_+0" / f"rseed{selection_seed}" / "hard_sseed42.json"
        soft_result = root / "results" / "lambda_+0" / f"rseed{selection_seed}" / "soft1_sseed42.json"
        for result in (hard_result, soft_result):
            payload = json.loads(result.read_text(encoding="utf-8"))
            if payload["train_manifest_sha256"] != file_sha256(manifest_path):
                raise RuntimeError("existing lambda0 result did not use the audited manifest")
        manifests[str(selection_seed)] = {
            "path": str(manifest_path.resolve()),
            "sha256": file_sha256(manifest_path),
        }
        seed_audits[str(selection_seed)] = {
            "low_entropy_half": group_metrics(low),
            "high_entropy_half": group_metrics(high),
            "holdout_entropy_high_minus_low": statistics.mean(
                row["holdout_entropy"] for row in high
            )
            - statistics.mean(row["holdout_entropy"] for row in low),
            "per_class": per_class,
            "low_relative_paths": sorted(row["relative_path"] for row in low),
            "high_relative_paths": sorted(row["relative_path"] for row in high),
        }

    top_rows = []
    for class_id in range(CLASSES):
        class_rows = [row for row in rows if int(row["class_id"]) == class_id]
        chosen = sorted(
            class_rows,
            key=lambda row: (-float(row["calibration_entropy"]), row["relative_path"]),
        )[:IPC]
        for slot, row in enumerate(chosen):
            top_rows.append({**row, "slot": slot, "selection_score": row["calibration_entropy"]})
    split_path = Path(base["official_split_file_list"])
    top_manifest = {
        "status": "complete",
        "experiment": "imagenette_highest_entropy_top10_v1",
        "dataset": "imagenet-nette",
        "classes": CLASSES,
        "ipc": IPC,
        "lambda": "highest_entropy_top10_per_class",
        "selection_seed": None,
        "selection_method": "deterministic_highest_calibration_entropy_top10_within_true_class",
        "selection_images": len(top_rows),
        "teacher_checkpoint": base["teacher_checkpoint"],
        "teacher_checkpoint_sha256": base["teacher_checkpoint_sha256"],
        "calibration_statistics_source": str(per_image_path.resolve()),
        "calibration_statistics_source_sha256": file_sha256(per_image_path),
        "training_sample_weighting": "equal",
        "candidate_pool": "all images of the true class; no Teacher correctness/confidence filter",
        "official_split_file_list": str(split_path.resolve()),
        "official_split_file_list_sha256": base["official_split_file_list_sha256"],
        "class_to_idx": train_dataset.class_to_idx,
        "images": top_rows,
    }
    top_manifest_path = root / "manifests" / "highest_entropy_top10_per_class.json"
    atomic_json(top_manifest_path, top_manifest)

    payload = {
        "status": "complete_pending_allocation_review",
        "experiment": "imagenette_entropy_allocation_v1",
        "parent_preflight": str(base_path.resolve()),
        "parent_preflight_sha256": file_sha256(base_path),
        "per_image_statistics": str(per_image_path.resolve()),
        "per_image_statistics_sha256": file_sha256(per_image_path),
        "lambda0_manifests": manifests,
        "lambda0_seed_audits": seed_audits,
        "allocation_invariants": {
            "soft_images_per_intermediate_condition": 50,
            "images_per_class_per_group": 5,
            "same_images_across_four_conditions": True,
            "same_student_initialization_shuffle_and_augmentation_keys": True,
            "test_evaluation_has_no_random_transform_or_training_rng_consumption": True,
        },
        "highest_entropy_top10_manifest": str(top_manifest_path.resolve()),
        "highest_entropy_top10_manifest_sha256": file_sha256(top_manifest_path),
        "highest_entropy_top10_metrics": group_metrics(top_rows),
        "training_gate": str(root / "preflight" / "lambda0_entropy_allocation_gate.json"),
    }
    atomic_json(output, payload)
    print(json.dumps({"status": payload["status"], "lambda0_manifests": 3, "top10_images": len(top_rows)}))


if __name__ == "__main__":
    main()
