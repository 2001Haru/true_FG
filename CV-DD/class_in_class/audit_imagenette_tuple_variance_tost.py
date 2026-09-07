"""Decompose the 3x3 tuple variance and plan manifest counts for TOST."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.stats import chi2, norm, t

from imagenette_entropy_protocol import atomic_json, file_sha256


def two_way_components(matrix):
    rows, columns = matrix.shape
    grand = matrix.mean()
    row_means = matrix.mean(axis=1)
    column_means = matrix.mean(axis=0)
    row_ss = columns * np.square(row_means - grand).sum()
    column_ss = rows * np.square(column_means - grand).sum()
    residual_ss = np.square(
        matrix - row_means[:, None] - column_means[None, :] + grand
    ).sum()
    row_ms = row_ss / (rows - 1)
    column_ms = column_ss / (columns - 1)
    residual_ms = residual_ss / ((rows - 1) * (columns - 1))
    manifest_variance = max((row_ms - residual_ms) / columns, 0.0)
    student_variance = max((column_ms - residual_ms) / rows, 0.0)
    residual_variance = residual_ms
    total = manifest_variance + student_variance + residual_variance
    one_way_between_ms = columns * np.square(row_means - grand).sum() / (rows - 1)
    one_way_within_ms = np.square(matrix - row_means[:, None]).sum() / (
        rows * (columns - 1)
    )
    one_way_manifest_variance = max(
        (one_way_between_ms - one_way_within_ms) / columns, 0.0
    )
    return {
        "matrix": matrix.tolist(),
        "grand_mean": float(grand),
        "manifest_means": row_means.tolist(),
        "student_seed_means": column_means.tolist(),
        "two_way_random_components": {
            "manifest_variance": manifest_variance,
            "student_seed_variance": student_variance,
            "manifest_by_student_residual_variance": residual_variance,
            "variance_shares": {
                "manifest": manifest_variance / total,
                "student_seed": student_variance / total,
                "manifest_by_student_residual": residual_variance / total,
            },
        },
        "one_way_manifest_random_intercept": {
            "manifest_variance": one_way_manifest_variance,
            "within_manifest_student_residual_variance": one_way_within_ms,
            "icc": one_way_manifest_variance
            / (one_way_manifest_variance + one_way_within_ms),
        },
    }


def tost_power(sample_size, true_mean, sigma, bound):
    df = sample_size - 1
    critical = float(t.ppf(0.95, df))
    mean_se = sigma / math.sqrt(sample_size)
    lower = float(chi2.ppf(1e-9, df))
    upper = float(chi2.ppf(1 - 1e-9, df))

    def integrand(value):
        half_width = critical * sigma * math.sqrt(value / df / sample_size)
        remaining = bound - half_width
        if remaining <= 0:
            return 0.0
        probability = norm.cdf((remaining - true_mean) / mean_se) - norm.cdf(
            (-remaining - true_mean) / mean_se
        )
        return probability * chi2.pdf(value, df)

    return float(quad(integrand, lower, upper, epsabs=1e-7, limit=100)[0])


def required_manifests(target_power, true_mean, sigma, bound):
    low, high = 2, 4
    while tost_power(high, true_mean, sigma, bound) < target_power:
        high *= 2
        if high > 100000:
            raise RuntimeError("TOST manifest requirement exceeds search limit")
    while low + 1 < high:
        middle = (low + high) // 2
        if tost_power(middle, true_mean, sigma, bound) >= target_power:
            high = middle
        else:
            low = middle
    return high


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--checkpoint-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    audit = json.loads(args.checkpoint_audit.read_text(encoding="utf-8"))
    rows = audit["checkpoint_readout"]["tuples"]
    matrices = {
        key: np.zeros((3, 3), dtype=np.float64)
        for key in (
            "high_nll_identity_value",
            "low_nll_identity_value",
            "mass_scaled_nll_residual",
            "high_accuracy_identity_value",
            "low_accuracy_identity_value",
        )
    }
    for row_index, selection_seed in enumerate((0, 1, 2)):
        for column_index, student_seed in enumerate((42, 43, 44)):
            high = next(
                row
                for row in rows
                if row["group"] == "high"
                and row["selection_seed"] == selection_seed
                and row["student_seed"] == student_seed
            )
            low = next(
                row
                for row in rows
                if row["group"] == "low"
                and row["selection_seed"] == selection_seed
                and row["student_seed"] == student_seed
            )
            high_result = json.loads(
                (
                    root
                    / "results/lambda0_entropy_allocation"
                    / f"rseed{selection_seed}"
                    / f"high_entropy_soft1_sseed{student_seed}.json"
                ).read_text(encoding="utf-8")
            )
            low_result = json.loads(
                (
                    root
                    / "results/lambda0_entropy_allocation"
                    / f"rseed{selection_seed}"
                    / f"low_entropy_soft1_sseed{student_seed}.json"
                ).read_text(encoding="utf-8")
            )
            high_mass = 1.0 - high_result["teacher_online_label_statistics"][
                "mean_true_class_probability"
            ]
            low_mass = 1.0 - low_result["teacher_online_label_statistics"][
                "mean_true_class_probability"
            ]
            high_nll = high["permuted"]["all"]["nll"] - high["original"]["all"]["nll"]
            low_nll = low["permuted"]["all"]["nll"] - low["original"]["all"]["nll"]
            matrices["high_nll_identity_value"][row_index, column_index] = high_nll
            matrices["low_nll_identity_value"][row_index, column_index] = low_nll
            matrices["mass_scaled_nll_residual"][row_index, column_index] = (
                high_nll - high_mass / low_mass * low_nll
            )
            matrices["high_accuracy_identity_value"][row_index, column_index] = (
                high["original"]["all"]["accuracy"]
                - high["permuted"]["all"]["accuracy"]
            )
            matrices["low_accuracy_identity_value"][row_index, column_index] = (
                low["original"]["all"]["accuracy"]
                - low["permuted"]["all"]["accuracy"]
            )
    decompositions = {
        key: two_way_components(matrix) for key, matrix in matrices.items()
    }
    primary = decompositions["mass_scaled_nll_residual"]
    components = primary["two_way_random_components"]
    # Planning is conditional on retaining the standard three Student seeds as
    # fixed blocks. A future manifest mean therefore has variance sigma_M^2 +
    # sigma_E^2 / 3; the shared Student main effect is removed by blocking.
    manifest_mean_sigma = math.sqrt(
        components["manifest_variance"]
        + components["manifest_by_student_residual_variance"] / 3
    )
    reference_effect = float(matrices["high_nll_identity_value"].mean())
    observed_residual = float(matrices["mass_scaled_nll_residual"].mean())
    planning = {}
    for fraction in (0.25, 0.50, 1.00):
        bound = reference_effect * fraction
        planning[str(fraction)] = {
            "equivalence_bound": bound,
            "best_case_true_residual_zero": {
                "manifests_for_80_percent_power": required_manifests(
                    0.80, 0.0, manifest_mean_sigma, bound
                ),
                "manifests_for_90_percent_power": required_manifests(
                    0.90, 0.0, manifest_mean_sigma, bound
                ),
            },
            "if_true_residual_equals_current_point_estimate": {
                "assumed_true_residual": observed_residual,
                "manifests_for_80_percent_power": required_manifests(
                    0.80, observed_residual, manifest_mean_sigma, bound
                ),
                "manifests_for_90_percent_power": required_manifests(
                    0.90, observed_residual, manifest_mean_sigma, bound
                ),
            },
        }
    current_manifest_means = np.asarray(primary["manifest_means"])
    current_sigma = float(current_manifest_means.std(ddof=1))
    current_half_width = float(t.ppf(0.95, 2) * current_sigma / math.sqrt(3))
    payload = {
        "status": "complete",
        "experiment": "imagenette_tuple_variance_and_tost_planning",
        "design": "3 selection manifests x the same 3 Student seeds; one observation per cell",
        "decompositions": decompositions,
        "primary_tost_estimand": "high NLL identity value - online non-target-mass ratio x low NLL identity value",
        "planning_assumptions": {
            "normal_manifest_means": True,
            "alpha_each_one_sided": 0.05,
            "power_is_joint_TOST_power_at_assumed_true_residual": True,
            "student_seeds": "42/43/44 retained as fixed crossed blocks for every new manifest",
            "manifest_mean_sigma": manifest_mean_sigma,
            "variance_formula": "sigma_manifest^2 + sigma_manifest_by_student^2 / 3",
            "reference_high_group_nll_identity_value": reference_effect,
            "observed_scaled_residual": observed_residual,
        },
        "current_three_manifest_90_percent_ci": [
            observed_residual - current_half_width,
            observed_residual + current_half_width,
        ],
        "tost_manifest_requirements": planning,
        "checkpoint_audit": str(args.checkpoint_audit.resolve()),
        "checkpoint_audit_sha256": file_sha256(args.checkpoint_audit.resolve()),
    }
    atomic_json(args.output, payload)
    print(json.dumps({"status": "complete", "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
