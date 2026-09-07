"""Audit the existing A-versus-Aprime gap over all saved evaluation times."""

import argparse
import json
import math
import os
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from summarize_imagenette_ordering_hierarchy import PAIRS, block_adjusted
from train_imagenette_ordering_hierarchy import EVAL_EPOCHS


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    output = args.output_root.resolve()
    histories = {}
    for manifest_seed, students in PAIRS.items():
        for student_seed in students:
            histories[(manifest_seed, student_seed)] = {}
            for arm in ("A", "Aprime"):
                path = root / f"results/rseed{manifest_seed}/sseed{student_seed}/{arm}.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                histories[(manifest_seed, student_seed)][arm] = {
                    int(row["epoch"]): row for row in payload["test_history"]
                }
    timepoints = []
    manifest_trajectories = {seed: [] for seed in PAIRS}
    for epoch in EVAL_EPOCHS:
        rows = []
        for manifest_seed, students in PAIRS.items():
            nll = []
            top1 = []
            for student_seed in students:
                pair = histories[(manifest_seed, student_seed)]
                nll.append(pair["Aprime"][epoch]["test_loss"] - pair["A"][epoch]["test_loss"])
                top1.append(pair["Aprime"][epoch]["test_top1"] - pair["A"][epoch]["test_top1"])
            contrasts = {
                "Aprime_minus_A_nll": statistics.mean(nll),
                "Aprime_minus_A_top1": statistics.mean(top1),
            }
            rows.append({
                "manifest_seed": manifest_seed, "student_pair": list(students),
                "contrasts": contrasts,
            })
            manifest_trajectories[manifest_seed].append({"epoch": epoch, **contrasts})
        nll = block_adjusted(rows, "Aprime_minus_A_nll")
        top1 = block_adjusted(rows, "Aprime_minus_A_top1")
        timepoints.append({
            "epoch": epoch, "updates": 2 * epoch,
            "Aprime_minus_A_nll": nll,
            "Aprime_minus_A_top1": top1,
            "manifest_nll_negative": sum(row["contrasts"]["Aprime_minus_A_nll"] < 0 for row in rows),
            "manifest_top1_positive": sum(row["contrasts"]["Aprime_minus_A_top1"] > 0 for row in rows),
        })
    updates = np.asarray([0] + [row["updates"] for row in timepoints], dtype=np.float64)
    damage = np.asarray([0.0] + [-row["Aprime_minus_A_nll"]["mean"] for row in timepoints])
    def fit_through_origin(predictor):
        coefficient = float(np.dot(predictor, damage) / np.dot(predictor, predictor))
        fitted = coefficient * predictor
        sse = float(np.square(damage - fitted).sum())
        sst0 = float(np.square(damage).sum())
        return {"coefficient": coefficient, "zero_origin_r2": 1.0 - sse / sst0, "sse": sse}
    adjacent = np.diff(damage)
    trajectory_summary = {
        "damage_definition": "L_A-L_Aprime = -(L_Aprime-L_A)",
        "adjacent_increases": int((adjacent > 0).sum()),
        "adjacent_decreases": int((adjacent < 0).sum()),
        "first_observed_damage_at_400_updates": float(damage[1]),
        "damage_at_1200_updates": float(damage[3]),
        "final_damage_at_4000_updates": float(damage[-1]),
        "fraction_of_final_already_present_at_400_updates": float(damage[1] / damage[-1]),
        "fraction_of_final_already_present_at_1200_updates": float(damage[3] / damage[-1]),
        "spearman_updates_vs_damage": {
            "rho": float(spearmanr(updates[1:], damage[1:]).statistic),
            "p": float(spearmanr(updates[1:], damage[1:]).pvalue),
        },
        "through_origin_fits_including_update0": {
            "sqrt_updates": fit_through_origin(np.sqrt(updates)),
            "linear_updates": fit_through_origin(updates),
        },
    }
    payload = {
        "status": "complete", "no_student_retraining": True,
        "independent_unit": "nine manifests; average two Students; Student pair fixed block",
        "timepoints": timepoints, "manifest_trajectories": manifest_trajectories,
        "trajectory_summary": trajectory_summary,
    }
    atomic_json(output / "ordering_gap_trajectory.json", payload)

    epochs = np.asarray([row["epoch"] for row in timepoints])
    mean = np.asarray([-row["Aprime_minus_A_nll"]["mean"] for row in timepoints])
    lower = np.asarray([-row["Aprime_minus_A_nll"]["ci95"][1] for row in timepoints])
    upper = np.asarray([-row["Aprime_minus_A_nll"]["ci95"][0] for row in timepoints])
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.axhline(0, color="0.5", linewidth=1)
    axis.fill_between(2 * epochs, lower, upper, color="#4477AA", alpha=0.2, label="blocked 95% CI")
    axis.plot(2 * epochs, mean, "o-", color="#225588", linewidth=2, markersize=4, label=r"$L_A-L_{A'}$")
    for milestone in (2666, 3332):
        axis.axvline(milestone, color="#BB5566", linestyle="--", linewidth=1)
    axis.set(xlabel="Optimizer updates", ylabel="Final-test NLL gap at checkpoint", title="A damage relative to equidistant A′ over training")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "ordering_gap_trajectory.png", dpi=180)
    fig.savefig(output / "ordering_gap_trajectory.pdf")
    plt.close(fig)
    print(json.dumps({"status": "complete", "output": str(output), **trajectory_summary}, indent=2))


if __name__ == "__main__":
    main()
