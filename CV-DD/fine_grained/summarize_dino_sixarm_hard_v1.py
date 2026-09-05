"""Combine the frozen five-arm matrix with the shell-random extension."""

import argparse
import copy
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
SELECTION_SEEDS = (0, 1, 2)
STUDENT_SEEDS = (42, 43, 44)


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
    parser.add_argument("--base-summary", required=True, type=Path)
    parser.add_argument("--expected-base-summary-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    errors = []
    actual_base_hash = sha256(args.base_summary)
    if actual_base_hash != args.expected_base_summary_sha256:
        errors.append(
            f"base summary hash {actual_base_hash} != {args.expected_base_summary_sha256}"
        )
    base = json.loads(args.base_summary.read_text(encoding="utf-8"))
    if base.get("status") != "complete" or base.get("found_results") != 99:
        errors.append("base five-arm summary is incomplete")
    combined = copy.deepcopy(base["summary"])
    found_new = 0
    for dataset, (classes, validation_images) in DATASETS.items():
        extension_audit_path = (
            args.experiment_root / "selection_extension_audits" /
            f"{dataset}_shell_random.json"
        )
        if not extension_audit_path.is_file():
            errors.append(f"missing extension audit: {extension_audit_path}")
            extension_hash = None
            extension_audit = {}
        else:
            extension_hash = sha256(extension_audit_path)
            extension_audit = json.loads(extension_audit_path.read_text(encoding="utf-8"))
            expected_audit = {
                "status": "complete",
                "experiment": "dino_sixarm_ipc1",
                "selection_method": "shell_random",
                "dataset": dataset,
                "classes": classes,
                "ipc": 1,
                "selection_seeds": list(SELECTION_SEEDS),
                "uses_rival_similarity": False,
            }
            for key, value in expected_audit.items():
                if extension_audit.get(key) != value:
                    errors.append(f"{extension_audit_path}: {key} mismatch")
        rows = []
        for selection_seed in SELECTION_SEEDS:
            arm = f"shell_random_rseed{selection_seed}"
            for student_seed in STUDENT_SEEDS:
                result = (
                    args.experiment_root / "results" / dataset / arm /
                    f"sseed{student_seed}.json"
                )
                if not result.is_file():
                    errors.append(f"missing result: {result}")
                    continue
                payload = json.loads(result.read_text(encoding="utf-8"))
                found_new += 1
                expected = {
                    "status": "complete",
                    "protocol": "hard_label_v1",
                    "method": "DINORealSelection",
                    "dataset": dataset,
                    "ipc": 1,
                    "selection_experiment": "dino_sixarm_ipc1",
                    "selection_method": "shell_random",
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
                    "selection_audit_sha256": extension_hash,
                }
                for key, value in expected.items():
                    if not same(payload.get(key), value):
                        errors.append(f"{result}: {key} mismatch")
                manifest = Path(payload.get("selection_manifest", ""))
                if not manifest.is_file() or sha256(manifest) != payload.get(
                    "selection_manifest_sha256"
                ):
                    errors.append(f"{result}: manifest missing or changed")
                rows.append(
                    {
                        "selection_seed": selection_seed,
                        "student_seed": student_seed,
                        "final_top1": float(payload["final_top1"]),
                        "best_top1_diagnostic": float(payload["best_top1_diagnostic"]),
                        "best_update_diagnostic": int(payload["best_update_diagnostic"]),
                        "path": str(result.resolve()),
                    }
                )
        if len(rows) == 9:
            final = [row["final_top1"] for row in rows]
            best = [row["best_top1_diagnostic"] for row in rows]
            shell_random = {
                "runs": rows,
                "primary_final_mean": statistics.mean(final),
                "primary_final_sample_std": statistics.stdev(final),
                "diagnostic_best_mean": statistics.mean(best),
                "diagnostic_best_sample_std": statistics.stdev(best),
                "final_mean_by_selection_seed": {
                    str(seed): statistics.mean(
                        row["final_top1"] for row in rows if row["selection_seed"] == seed
                    )
                    for seed in SELECTION_SEEDS
                },
                "selection_audit": str(extension_audit_path.resolve()),
                "selection_audit_sha256": extension_hash,
                "overlap_counts_with_existing_arms": extension_audit.get(
                    "overlap_counts_with_existing_arms"
                ),
            }
            global_random = {
                (row["selection_seed"], row["student_seed"]): row["final_top1"]
                for row in combined[dataset]["random"]["runs"]
            }
            paired = [
                row["final_top1"]
                - global_random[(row["selection_seed"], row["student_seed"])]
                for row in rows
            ]
            shell_selection_means = shell_random["final_mean_by_selection_seed"]
            global_selection_means = {
                str(seed): statistics.mean(
                    row["final_top1"]
                    for row in combined[dataset]["random"]["runs"]
                    if row["selection_seed"] == seed
                )
                for seed in SELECTION_SEEDS
            }
            shell_random["comparison_vs_global_random"] = {
                "design": (
                    "selection-seed and Student-seed indices aligned, but selection hashes use "
                    "independent namespaces; not a common-random-number paired selection test"
                ),
                "mean": statistics.mean(paired),
                "sample_std": statistics.stdev(paired),
                "wins": sum(value > 0 for value in paired),
                "ties": sum(value == 0 for value in paired),
                "losses": sum(value < 0 for value in paired),
                "values": paired,
                "shell_random_final_mean_by_selection_seed": shell_selection_means,
                "global_random_final_mean_by_selection_seed": global_selection_means,
                "selection_seed_mean_deltas": {
                    str(seed): shell_selection_means[str(seed)] - global_selection_means[str(seed)]
                    for seed in SELECTION_SEEDS
                },
            }
            combined[dataset]["shell_random"] = shell_random
    payload = {
        "status": "complete" if found_new == 27 and not errors else "incomplete",
        "protocol": "hard_label_v1",
        "experiment": "dino_sixarm_ipc1",
        "primary_metric": "final_update_top1",
        "base_fivearm_summary": str(args.base_summary.resolve()),
        "base_fivearm_summary_sha256": actual_base_hash,
        "expected_results": 126,
        "found_results": 99 + found_new,
        "shell_random_selection_seeds": list(SELECTION_SEEDS),
        "shell_random_student_seeds": list(STUDENT_SEEDS),
        "errors": errors,
        "summary": combined,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "complete":
        raise RuntimeError(f"six-arm matrix incomplete: {len(errors)} errors")


if __name__ == "__main__":
    main()
