"""Audit and summarize the 3 datasets x 3 IPC x 3x3 seed CoDA matrix."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--feature-space", required=True, choices=("vae", "dinov2"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    errors = []
    summary = {}
    found = 0
    for dataset, (classes, validation_images) in DATASETS.items():
        summary[dataset] = {}
        for ipc in (1, 3, 5):
            rows = []
            for generation_seed in (0, 1, 2):
                for student_seed in (42, 43, 44):
                    path = (
                        args.experiment_root / "results" / dataset /
                        f"ipc{ipc}_gseed{generation_seed}_sseed{student_seed}.json"
                    )
                    if not path.is_file():
                        errors.append(f"missing: {path}")
                        continue
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    found += 1
                    expected = {
                        "method": "CoDA",
                        "coda_feature_space": args.feature_space,
                        "supervision": "hard_label_cross_entropy",
                        "training_target": "hard_coarse_label",
                        "student_initialization": "random",
                        "student_seed": student_seed,
                        "epochs": 400,
                        "batch_size": 14 if dataset == "SC_imsize224" else 20,
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
                        "generation_seed": generation_seed,
                    }
                    for key, value in expected.items():
                        actual = payload.get(key)
                        if isinstance(value, float):
                            valid = actual is not None and math.isclose(float(actual), value, abs_tol=1e-12)
                        else:
                            valid = actual == value
                        if not valid:
                            errors.append(f"{path}: {key}={actual!r}, expected {value!r}")
                    for key in ("generation_audit_sha256", "generation_config_sha256", "synthetic_tree_sha256"):
                        if not payload.get(key):
                            errors.append(f"{path}: missing {key}")
                    rows.append(
                        {
                            "generation_seed": generation_seed,
                            "student_seed": student_seed,
                            "best_top1": float(payload["best_top1"]),
                            "final_top1": float(payload["final_epoch_top1"]),
                            "path": str(path.resolve()),
                        }
                    )
            if len(rows) == 9:
                best = [row["best_top1"] for row in rows]
                final = [row["final_top1"] for row in rows]
                generation_means = {
                    str(seed): statistics.mean(
                        row["best_top1"] for row in rows if row["generation_seed"] == seed
                    )
                    for seed in (0, 1, 2)
                }
                summary[dataset][str(ipc)] = {
                    "runs": rows,
                    "best_mean": statistics.mean(best),
                    "best_sample_std": statistics.stdev(best),
                    "final_mean": statistics.mean(final),
                    "final_sample_std": statistics.stdev(final),
                    "best_mean_by_generation_seed": generation_means,
                    "generation_seed_mean_sample_std": statistics.stdev(generation_means.values()),
                }
    payload = {
        "status": "complete" if found == 81 and not errors else "incomplete",
        "method": "CoDA",
        "feature_space": args.feature_space,
        "supervision": "hard_label_cross_entropy",
        "generation_seeds": [0, 1, 2],
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
        raise RuntimeError(f"CoDA matrix incomplete: found={found}, errors={len(errors)}")


if __name__ == "__main__":
    main()
