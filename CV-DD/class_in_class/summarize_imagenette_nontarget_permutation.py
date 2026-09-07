"""Summarize the paired non-target identity permutation intervention."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    values = {key: [] for key in (
        "original_high_soft", "permuted_high_soft", "original_low_soft", "permuted_low_soft",
        "high_permutation_penalty", "low_permutation_penalty", "penalty_high_minus_low",
        "permuted_high_minus_low", "permuted_high_minus_hard", "permuted_low_minus_hard",
    )}
    rows = []
    errors = []
    invariants = []
    for rseed in (0, 1, 2):
        for sseed in (42, 43, 44):
            paths = {
                "hard": root / "results/lambda_+0" / f"rseed{rseed}" / f"hard_sseed{sseed}.json",
                "original_high": root / "results/lambda0_entropy_allocation" / f"rseed{rseed}" / f"high_entropy_soft1_sseed{sseed}.json",
                "original_low": root / "results/lambda0_entropy_allocation" / f"rseed{rseed}" / f"low_entropy_soft1_sseed{sseed}.json",
                "permuted_high": root / "results/lambda0_nontarget_permutation" / f"rseed{rseed}" / f"high_entropy_permuted_soft1_sseed{sseed}.json",
                "permuted_low": root / "results/lambda0_nontarget_permutation" / f"rseed{rseed}" / f"low_entropy_permuted_soft1_sseed{sseed}.json",
            }
            if any(not path.is_file() for path in paths.values()):
                errors.append(f"missing tuple r{rseed}/s{sseed}")
                continue
            p = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
            if len({item["train_manifest_sha256"] for item in p.values()}) != 1:
                errors.append(f"manifest mismatch r{rseed}/s{sseed}")
            if len({item["initial_student_state_sha256"] for item in p.values()}) != 1:
                errors.append(f"Student initialization mismatch r{rseed}/s{sseed}")
            a = {name: float(item["final_top1"]) for name, item in p.items()}
            row = {
                "selection_seed": rseed,
                "student_seed": sseed,
                **a,
                "high_permutation_penalty": a["original_high"] - a["permuted_high"],
                "low_permutation_penalty": a["original_low"] - a["permuted_low"],
                "penalty_high_minus_low": (a["original_high"] - a["permuted_high"]) - (a["original_low"] - a["permuted_low"]),
                "permuted_high_minus_low": a["permuted_high"] - a["permuted_low"],
                "permuted_high_minus_hard": a["permuted_high"] - a["hard"],
                "permuted_low_minus_hard": a["permuted_low"] - a["hard"],
            }
            rows.append(row)
            values["original_high_soft"].append(a["original_high"])
            values["permuted_high_soft"].append(a["permuted_high"])
            values["original_low_soft"].append(a["original_low"])
            values["permuted_low_soft"].append(a["permuted_low"])
            for key in values:
                if key in row:
                    values[key].append(row[key])
            invariants.extend((p["permuted_high"]["permutation_invariant_audit"], p["permuted_low"]["permutation_invariant_audit"]))
    found = len(rows) * 2
    payload = {
        "status": "complete" if found == 18 and not errors else "incomplete",
        "experiment": "imagenette_nontarget_permutation_v1",
        "expected_new_results": 18,
        "found_new_results": found,
        "errors": errors,
        "summary": {key: stats(item) for key, item in values.items()},
        "tuples": rows,
        "global_invariant_audit": {
            "result_files": len(invariants),
            "views": sum(item["views"] for item in invariants),
            "argmax_correctness_mismatches": sum(item["argmax_correctness_mismatches"] for item in invariants),
            **{
                key: max(item[key] for item in invariants)
                for key in (
                    "max_abs_entropy_delta", "max_abs_true_probability_delta",
                    "max_abs_maximum_probability_delta", "max_abs_onehot_l1_delta",
                    "max_abs_onehot_l2_delta",
                )
            },
        },
        "primary_mechanism_contrast": "(original_high-permuted_high)-(original_low-permuted_low)",
        "primary_metric": "Final Top-1 at update4000",
    }
    atomic_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("status", "expected_new_results", "found_new_results", "errors")}, indent=2))
    if payload["status"] != "complete":
        raise RuntimeError("non-target permutation matrix incomplete")


if __name__ == "__main__":
    main()

