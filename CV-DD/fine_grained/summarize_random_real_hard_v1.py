"""Audit and summarize the 81-result random-real hard-label v1 matrix."""

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

from hard_label_v1_protocol import (
    BACKBONE_LR,
    BACKBONE_MIN_LR,
    EVAL_EVERY_UPDATES,
    HEAD_LR,
    HEAD_MIN_LR,
    MOMENTUM,
    PHYSICAL_BATCH_SIZE,
    PROTOCOL_NAME,
    TOTAL_UPDATES,
    TRAIN_WORKERS,
    VALIDATION_BATCH_SIZE,
    WEIGHT_DECAY,
)


DATASETS = {
    "CUB_imsize224": (200, 5794),
    "A_imsize224": (100, 3333),
    "SC_imsize224": (196, 8041),
}


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    errors = []
    found = 0
    summary = {}
    selection_sources = {}
    for dataset, (classes, validation_images) in DATASETS.items():
        summary[dataset] = {}
        for ipc in (1, 3, 5):
            rows = []
            for selection_seed in (0, 1, 2):
                for student_seed in (42, 43, 44):
                    result = (
                        args.experiment_root / "results" / dataset /
                        f"ipc{ipc}_rseed{selection_seed}_sseed{student_seed}.json"
                    )
                    if not result.is_file():
                        errors.append(f"missing result: {result}")
                        continue
                    payload = json.loads(result.read_text(encoding="utf-8"))
                    found += 1
                    expected = {
                        "status": "complete",
                        "method": "RandomReal",
                        "protocol": PROTOCOL_NAME,
                        "dataset": dataset,
                        "ipc": ipc,
                        "selection_seed": selection_seed,
                        "student_seed": student_seed,
                        "student_initialization": "imagenet1k_v1",
                        "training_target": "hard_coarse_label",
                        "optimizer": "sgd",
                        "momentum": MOMENTUM,
                        "weight_decay": WEIGHT_DECAY,
                        "backbone_initial_lr": BACKBONE_LR,
                        "head_initial_lr": HEAD_LR,
                        "backbone_min_lr": BACKBONE_MIN_LR,
                        "head_min_lr": HEAD_MIN_LR,
                        "total_optimizer_updates": TOTAL_UPDATES,
                        "updates_completed": TOTAL_UPDATES,
                        "gradient_accumulation_steps": 1,
                        "physical_batch_size": PHYSICAL_BATCH_SIZE,
                        "dataloader_workers": TRAIN_WORKERS,
                        "persistent_workers": True,
                        "validation_batch_size": VALIDATION_BATCH_SIZE,
                        "evaluation_every_updates": EVAL_EVERY_UPDATES,
                        "num_classes": classes,
                        "train_images": classes * ipc,
                        "validation_images": validation_images,
                        "primary_metric": "final_update_top1",
                        "eval_openblas_num_threads": 1,
                    }
                    for key, value in expected.items():
                        if not same(payload.get(key), value):
                            errors.append(
                                f"{result}: {key}={payload.get(key)!r}, expected {value!r}"
                            )
                    for key in (
                        "protocol_spec_sha256",
                        "selection_manifest_sha256",
                        "selected_tree_sha256",
                        "imagenet_initial_state_sha256",
                        "initial_head_sha256",
                        "final_checkpoint_sha256",
                    ):
                        if not payload.get(key):
                            errors.append(f"{result}: missing {key}")
                    manifest_path = Path(payload.get("selection_manifest", ""))
                    if not manifest_path.is_file():
                        errors.append(f"{result}: missing manifest {manifest_path}")
                    else:
                        if sha256(manifest_path) != payload.get("selection_manifest_sha256"):
                            errors.append(f"{result}: manifest hash mismatch")
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        manifest_expected = {
                            "status": "complete",
                            "method": "random_real",
                            "dataset": dataset,
                            "classes": classes,
                            "ipc": ipc,
                            "selection_seed": selection_seed,
                            "selected_images": classes * ipc,
                            "selected_tree_sha256": payload.get("selected_tree_sha256"),
                        }
                        for key, value in manifest_expected.items():
                            if manifest.get(key) != value:
                                errors.append(
                                    f"{manifest_path}: {key}={manifest.get(key)!r}, expected {value!r}"
                                )
                        source_set = {row["source_path"] for row in manifest.get("images", [])}
                        key = (dataset, selection_seed, ipc)
                        previous = selection_sources.setdefault(key, source_set)
                        if previous != source_set:
                            errors.append(f"{result}: inconsistent selection across students")
                    rows.append(
                        {
                            "selection_seed": selection_seed,
                            "student_seed": student_seed,
                            "final_top1": float(payload["final_top1"]),
                            "best_top1_diagnostic": float(payload["best_top1_diagnostic"]),
                            "path": str(result.resolve()),
                        }
                    )
            if len(rows) == 9:
                final = [row["final_top1"] for row in rows]
                best = [row["best_top1_diagnostic"] for row in rows]
                selection_means = {
                    str(seed): statistics.mean(
                        row["final_top1"] for row in rows if row["selection_seed"] == seed
                    )
                    for seed in (0, 1, 2)
                }
                summary[dataset][str(ipc)] = {
                    "runs": rows,
                    "primary_final_mean": statistics.mean(final),
                    "primary_final_sample_std": statistics.stdev(final),
                    "diagnostic_best_mean": statistics.mean(best),
                    "diagnostic_best_sample_std": statistics.stdev(best),
                    "final_mean_by_selection_seed": selection_means,
                    "selection_seed_mean_sample_std": statistics.stdev(selection_means.values()),
                }
    for dataset in DATASETS:
        for selection_seed in (0, 1, 2):
            keys = [(dataset, selection_seed, ipc) for ipc in (1, 3, 5)]
            if all(key in selection_sources for key in keys):
                if not selection_sources[keys[0]] <= selection_sources[keys[1]] <= selection_sources[keys[2]]:
                    errors.append(f"non-nested selection: {dataset}, seed={selection_seed}")
    payload = {
        "status": "complete" if found == 81 and not errors else "incomplete",
        "protocol": PROTOCOL_NAME,
        "method": "RandomReal",
        "primary_metric": "final_update_top1",
        "expected_results": 81,
        "found_results": found,
        "selection_seeds": [0, 1, 2],
        "student_seeds": [42, 43, 44],
        "ipcs": [1, 3, 5],
        "errors": errors,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "complete":
        raise RuntimeError(f"hard-label v1 matrix incomplete: {len(errors)} errors")


if __name__ == "__main__":
    main()
