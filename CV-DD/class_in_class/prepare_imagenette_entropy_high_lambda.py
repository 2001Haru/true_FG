"""Extend the frozen ImageNette entropy selection to high positive lambdas.

This reuses the parent per-image calibration/holdout statistics and DINO cache;
it does not recompute views or alter any parent manifest/result.
"""

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

from imagenette_entropy_protocol import (
    BASE_EXPERIMENT_NAME,
    CLASSES,
    DATASET_NAME,
    DISPLAY_NAME,
    HIGH_EXPERIMENT_NAME,
    IPC,
    SELECTION_SEEDS,
    atomic_json,
    file_sha256,
    gumbel_noise,
    validate_official_split,
)
from prepare_imagenette_entropy_selection import contact_sheet, load_dino, selection_metrics


NEW_LAMBDAS = (8, 16, 32)
AUDIT_LAMBDAS = (4, 8, 16, 32)


def distribution(values):
    values = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def enrich_metrics(indices, features, targets, rows):
    payload = selection_metrics(indices, features, targets, rows)
    selected = [rows[index] for index in indices]
    payload.update(
        {
            "calibration_entropy_distribution": distribution(
                row["calibration_entropy"] for row in selected
            ),
            "holdout_entropy_distribution": distribution(
                row["holdout_entropy"] for row in selected
            ),
            "calibration_teacher_error_rate": 1.0
            - statistics.mean(row["calibration_argmax_correct"] for row in selected),
            "holdout_teacher_error_rate": 1.0
            - statistics.mean(row["holdout_argmax_correct"] for row in selected),
            "dino_own_centroid_similarity_mean": statistics.mean(
                row["dino_own_centroid_similarity"] for row in selected
            ),
            "dino_radial_cosine_distance_mean": statistics.mean(
                row["dino_radial_cosine_distance"] for row in selected
            ),
        }
    )
    return payload


def overlap_metrics(left, right, targets):
    left = set(left)
    right = set(right)
    per_class = []
    targets = np.asarray(targets)
    for class_id in range(CLASSES):
        class_pool = set(np.flatnonzero(targets == class_id).tolist())
        overlap = len((left & class_pool) & (right & class_pool)) / IPC
        per_class.append(overlap)
    return {
        "global": len(left & right) / len(left),
        "per_class": per_class,
        "per_class_mean": statistics.mean(per_class),
        "per_class_min": min(per_class),
        "per_class_max": max(per_class),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--dino-cache", required=True, type=Path)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.experiment_root.resolve()
    output = root / "preflight" / "high_lambda_extension_audit.json"
    if output.exists() and not args.force:
        raise RuntimeError(f"refusing to overwrite existing extension audit: {output}")
    base_preflight_path = root / "preflight" / "selection_preflight.json"
    per_image_path = root / "preflight" / "per_image_teacher_entropy_and_dino.json"
    base = json.loads(base_preflight_path.read_text(encoding="utf-8"))
    per_image = json.loads(per_image_path.read_text(encoding="utf-8"))
    if base.get("status") != "complete_pending_manual_visual_review":
        raise RuntimeError("parent preflight is not complete")
    if file_sha256(args.teacher_checkpoint.resolve()) != base["teacher_checkpoint_sha256"]:
        raise RuntimeError("Teacher differs from parent preflight")

    train_dataset, _ = validate_official_split(args.data_root.resolve())
    rows = per_image["images"]
    expected_paths = [
        Path(path).relative_to(args.data_root.resolve()).as_posix()
        for path, _ in train_dataset.samples
    ]
    if (
        len(rows) != len(train_dataset)
        or [row["relative_path"] for row in rows] != expected_paths
        or [int(row["class_id"]) for row in rows] != train_dataset.targets
    ):
        raise RuntimeError("parent per-image audit differs from frozen train split")
    features, dino_metadata = load_dino(
        args.dino_cache.resolve(), train_dataset, args.data_root.resolve()
    )
    targets = np.asarray(train_dataset.targets)
    percentile = np.asarray([row["entropy_percentile"] for row in rows])

    split_path = Path(base["official_split_file_list"])
    if file_sha256(split_path) != base["official_split_file_list_sha256"]:
        raise RuntimeError("official split provenance changed")

    selection_sets = {}
    arm_summaries = {}
    manifests = {}
    for selection_seed in SELECTION_SEEDS:
        old_manifest_path = root / "manifests" / f"lambda_{4:+d}_rseed{selection_seed}.json"
        old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
        old_paths = {row["relative_path"] for row in old_manifest["images"]}
        selection_sets[(4, selection_seed)] = {
            index for index, row in enumerate(rows) if row["relative_path"] in old_paths
        }
        noise = np.asarray(
            [gumbel_noise(selection_seed, row["relative_path"]) for row in rows]
        )
        for lambda_value in NEW_LAMBDAS:
            score = lambda_value * (2.0 * percentile - 1.0) + noise
            selected_indices = []
            selected_records = []
            for class_id in range(CLASSES):
                class_indices = np.flatnonzero(targets == class_id)
                chosen = sorted(
                    class_indices,
                    key=lambda index: (-float(score[index]), rows[index]["relative_path"]),
                )[:IPC]
                selected_indices.extend(chosen)
                for slot, index in enumerate(chosen):
                    selected_records.append(
                        {
                            **rows[index],
                            "slot": slot,
                            "selection_score": float(score[index]),
                            "gumbel_noise": float(noise[index]),
                        }
                    )
            arm = f"lambda_{lambda_value:+d}_rseed{selection_seed}"
            manifest = {
                "status": "complete",
                "experiment": HIGH_EXPERIMENT_NAME,
                "parent_experiment": BASE_EXPERIMENT_NAME,
                "dataset": DATASET_NAME,
                "classes": CLASSES,
                "ipc": IPC,
                "lambda": lambda_value,
                "selection_seed": selection_seed,
                "selection_method": "teacher_entropy_percentile_gumbel_topk",
                "selection_images": len(selected_records),
                "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
                "teacher_checkpoint_sha256": base["teacher_checkpoint_sha256"],
                "calibration_statistics_source": str(per_image_path.resolve()),
                "calibration_statistics_source_sha256": file_sha256(per_image_path),
                "training_sample_weighting": "equal",
                "candidate_pool": "all images of the true class; no Teacher correctness/confidence filter",
                "official_split_file_list": str(split_path.resolve()),
                "official_split_file_list_sha256": base["official_split_file_list_sha256"],
                "class_to_idx": train_dataset.class_to_idx,
                "images": selected_records,
            }
            manifest_path = root / "manifests" / f"{arm}.json"
            atomic_json(manifest_path, manifest)
            manifests[arm] = str(manifest_path.resolve())
            selected = set(map(int, selected_indices))
            selection_sets[(lambda_value, selection_seed)] = selected
            arm_summaries[arm] = enrich_metrics(
                selected_indices, features, targets, rows
            )
            contact_sheet(
                root / "preflight" / "high_lambda_contact_sheets" / f"{arm}.jpg",
                selected_records,
                f"{DISPLAY_NAME} IPC10 {arm}",
            )

    top_indices = []
    top_records = []
    for class_id in range(CLASSES):
        class_indices = np.flatnonzero(targets == class_id)
        chosen = sorted(
            class_indices,
            key=lambda index: (-rows[index]["calibration_entropy"], rows[index]["relative_path"]),
        )[:IPC]
        top_indices.extend(chosen)
        for slot, index in enumerate(chosen):
            top_records.append(
                {
                    **rows[index],
                    "slot": slot,
                    "selection_score": rows[index]["calibration_entropy"],
                }
            )
    top_set = set(map(int, top_indices))
    top_metrics = enrich_metrics(top_indices, features, targets, rows)
    contact_sheet(
        root / "preflight" / "high_lambda_contact_sheets" / "highest_entropy_top10_per_class.jpg",
        top_records,
        f"{DISPLAY_NAME} highest entropy Top10/class",
    )

    adjacent = {}
    top_overlap = {}
    for selection_seed in SELECTION_SEEDS:
        for left, right in zip(AUDIT_LAMBDAS[:-1], AUDIT_LAMBDAS[1:]):
            adjacent[f"rseed{selection_seed}:{left:+d}_vs_{right:+d}"] = overlap_metrics(
                selection_sets[(left, selection_seed)],
                selection_sets[(right, selection_seed)],
                targets,
            )
        for lambda_value in AUDIT_LAMBDAS:
            top_overlap[f"rseed{selection_seed}:lambda{lambda_value:+d}_vs_top10"] = overlap_metrics(
                selection_sets[(lambda_value, selection_seed)], top_set, targets
            )

    payload = {
        "status": "complete_pending_high_lambda_review",
        "experiment": HIGH_EXPERIMENT_NAME,
        "parent_preflight": str(base_preflight_path.resolve()),
        "parent_preflight_sha256": file_sha256(base_preflight_path),
        "per_image_statistics": str(per_image_path.resolve()),
        "per_image_statistics_sha256": file_sha256(per_image_path),
        "teacher_checkpoint_sha256": base["teacher_checkpoint_sha256"],
        "official_split_file_list_sha256": base["official_split_file_list_sha256"],
        "selection_lambdas": list(NEW_LAMBDAS),
        "selection_seeds": list(SELECTION_SEEDS),
        "audit_uses_independent_holdout_views": True,
        "arm_summaries": arm_summaries,
        "adjacent_lambda_overlap": adjacent,
        "overlap_with_highest_entropy_top10_per_class": top_overlap,
        "highest_entropy_top10_per_class": {
            "definition": "top 10 by calibration entropy within each true class; deterministic path tie-break",
            "metrics": top_metrics,
            "relative_paths": [row["relative_path"] for row in top_records],
        },
        "dino": {
            "cache": str(args.dino_cache.resolve()),
            "cache_sha256": file_sha256(args.dino_cache.resolve()),
            "metadata": dino_metadata,
            "role": "audit only",
        },
        "manifests": manifests,
        "training_gate": {
            "required": True,
            "approval_file": str(
                root / "preflight" / "high_lambda_extension_gate.json"
            ),
            "contact_sheets": str(
                root / "preflight" / "high_lambda_contact_sheets"
            ),
        },
    }
    atomic_json(output, payload)
    print(json.dumps({"status": payload["status"], "manifests": len(manifests)}))


if __name__ == "__main__":
    main()
