"""Audit and summarize the 3x3x3x3 random-real hard-label matrix."""

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path


DATASETS = {
    "CUB_imsize224": (200, 5794, 20),
    "A_imsize224": (100, 3333, 20),
    "SC_imsize224": (196, 8041, 14),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    errors = []
    summary = {}
    found = 0
    selection_sources = {}
    for dataset, (classes, validation_images, batch_size) in DATASETS.items():
        summary[dataset] = {}
        for ipc in (1, 3, 5):
            rows = []
            for selection_seed in (0, 1, 2):
                for student_seed in (42, 43, 44):
                    path = (
                        args.experiment_root / "results" / dataset /
                        f"ipc{ipc}_rseed{selection_seed}_sseed{student_seed}.json"
                    )
                    if not path.is_file():
                        errors.append(f"missing: {path}")
                        continue
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    found += 1
                    expected = {
                        "method": "RandomReal",
                        "supervision": "hard_label_cross_entropy",
                        "training_target": "hard_coarse_label",
                        "student_initialization": "random",
                        "student_seed": student_seed,
                        "epochs": 400,
                        "batch_size": batch_size,
                        "gradient_accumulation_steps": 2,
                        "temperature": 20.0,
                        "optimizer": "adamw",
                        "learning_rate": 1e-3,
                        "weight_decay": 1e-5,
                        "cosine_eta": 2.0,
                        "dataloader_workers": 8,
                        "persistent_workers": True,
                        "num_classes": classes,
                        "validation_images": validation_images,
                        "mix_type": None,
                        "fkd_path": None,
                        "selection_seed": selection_seed,
                        "eval_openblas_num_threads": 1,
                    }
                    for key, value in expected.items():
                        actual = payload.get(key)
                        if isinstance(value, float):
                            valid = actual is not None and math.isclose(
                                float(actual), value, abs_tol=1e-12
                            )
                        else:
                            valid = actual == value
                        if not valid:
                            errors.append(f"{path}: {key}={actual!r}, expected {value!r}")
                    for key in ("selection_manifest_sha256", "selected_tree_sha256"):
                        if not payload.get(key):
                            errors.append(f"{path}: missing {key}")
                    manifest_path = Path(payload.get("selection_manifest", ""))
                    if not manifest_path.is_file():
                        errors.append(f"{path}: missing selection manifest {manifest_path}")
                    else:
                        manifest_hash = sha256(manifest_path)
                        if manifest_hash != payload.get("selection_manifest_sha256"):
                            errors.append(f"{path}: selection manifest SHA-256 mismatch")
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
                        selection_key = (dataset, selection_seed, ipc)
                        sources = {row["source_path"] for row in manifest.get("images", [])}
                        previous = selection_sources.setdefault(selection_key, sources)
                        if previous != sources:
                            errors.append(f"{path}: inconsistent selection manifest across students")
                    rows.append(
                        {
                            "selection_seed": selection_seed,
                            "student_seed": student_seed,
                            "best_top1": float(payload["best_top1"]),
                            "final_top1": float(payload["final_epoch_top1"]),
                            "path": str(path.resolve()),
                        }
                    )
            if len(rows) == 9:
                best = [row["best_top1"] for row in rows]
                final = [row["final_top1"] for row in rows]
                selection_means = {
                    str(seed): statistics.mean(
                        row["best_top1"] for row in rows if row["selection_seed"] == seed
                    )
                    for seed in (0, 1, 2)
                }
                summary[dataset][str(ipc)] = {
                    "runs": rows,
                    "best_mean": statistics.mean(best),
                    "best_sample_std": statistics.stdev(best),
                    "final_mean": statistics.mean(final),
                    "final_sample_std": statistics.stdev(final),
                    "best_mean_by_selection_seed": selection_means,
                    "selection_seed_mean_sample_std": statistics.stdev(selection_means.values()),
                }
    for dataset in DATASETS:
        for selection_seed in (0, 1, 2):
            keys = [(dataset, selection_seed, ipc) for ipc in (1, 3, 5)]
            if all(key in selection_sources for key in keys):
                if not (
                    selection_sources[keys[0]]
                    <= selection_sources[keys[1]]
                    <= selection_sources[keys[2]]
                ):
                    errors.append(
                        f"non-nested selection: {dataset} selection_seed={selection_seed}"
                    )
    payload = {
        "status": "complete" if found == 81 and not errors else "incomplete",
        "method": "RandomReal",
        "supervision": "hard_label_cross_entropy",
        "selection_seeds": [0, 1, 2],
        "student_seeds": [42, 43, 44],
        "ipcs": [1, 3, 5],
        "expected_results": 81,
        "found_results": found,
        "errors": errors,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "complete":
        raise RuntimeError(f"Random-real matrix incomplete: found={found}, errors={len(errors)}")


if __name__ == "__main__":
    main()
