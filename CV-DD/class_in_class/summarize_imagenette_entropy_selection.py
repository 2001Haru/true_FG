"""Summarize paired Hard/Soft entropy-selection curves and interactions."""

import argparse
import json
import os
import statistics
from pathlib import Path

from imagenette_entropy_protocol import SELECTION_SEEDS, STUDENT_SEEDS, atomic_json


def stats(values):
    values = [float(value) for value in values]
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
    parser.add_argument("--lambdas", nargs="+", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    lambdas = sorted(set(args.lambdas))
    if 0 not in lambdas:
        raise RuntimeError("summary requires lambda=0 interaction baseline")
    errors = []
    found = 0
    runs = {}
    for lambda_value in lambdas:
        for selection_seed in SELECTION_SEEDS:
            for student_seed in STUDENT_SEEDS:
                for supervision in ("hard", "soft1"):
                    path = (
                        args.experiment_root
                        / "results"
                        / f"lambda_{lambda_value:+d}"
                        / f"rseed{selection_seed}"
                        / f"{supervision}_sseed{student_seed}.json"
                    )
                    if not path.is_file():
                        errors.append(f"missing result: {path}")
                        continue
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    found += 1
                    key = (lambda_value, selection_seed, student_seed, supervision)
                    runs[key] = {
                        "lambda": lambda_value,
                        "selection_seed": selection_seed,
                        "student_seed": student_seed,
                        "supervision": supervision,
                        "final_top1": float(payload["final_top1"]),
                        "best_top1_diagnostic": float(payload["best_top1_diagnostic"]),
                        "best_epoch_diagnostic": int(payload["best_epoch_diagnostic"]),
                        "initial_student_state_sha256": payload["initial_student_state_sha256"],
                        "train_manifest_sha256": payload["train_manifest_sha256"],
                        "path": str(path.resolve()),
                    }
    summary = {}
    baseline_gain = {}
    for selection_seed in SELECTION_SEEDS:
        for student_seed in STUDENT_SEEDS:
            hard = runs.get((0, selection_seed, student_seed, "hard"))
            soft = runs.get((0, selection_seed, student_seed, "soft1"))
            if hard and soft:
                baseline_gain[(selection_seed, student_seed)] = soft["final_top1"] - hard["final_top1"]
    for lambda_value in lambdas:
        lambda_rows = []
        paired_gain = []
        interaction = []
        for selection_seed in SELECTION_SEEDS:
            for student_seed in STUDENT_SEEDS:
                hard = runs.get((lambda_value, selection_seed, student_seed, "hard"))
                soft = runs.get((lambda_value, selection_seed, student_seed, "soft1"))
                if not hard or not soft:
                    continue
                if hard["initial_student_state_sha256"] != soft["initial_student_state_sha256"]:
                    errors.append(f"lambda={lambda_value},r={selection_seed},s={student_seed}: initial Student mismatch")
                if hard["train_manifest_sha256"] != soft["train_manifest_sha256"]:
                    errors.append(f"lambda={lambda_value},r={selection_seed},s={student_seed}: manifest mismatch")
                gain = soft["final_top1"] - hard["final_top1"]
                value = gain - baseline_gain[(selection_seed, student_seed)]
                paired_gain.append(gain)
                interaction.append(value)
                lambda_rows.extend((hard, soft))
        if len(lambda_rows) == 18:
            summary[str(lambda_value)] = {
                "hard_final": stats(row["final_top1"] for row in lambda_rows if row["supervision"] == "hard"),
                "soft1_final": stats(row["final_top1"] for row in lambda_rows if row["supervision"] == "soft1"),
                "paired_soft1_minus_hard": stats(paired_gain),
                "interaction_vs_lambda0": stats(interaction),
                "runs": lambda_rows,
            }
    expected = len(lambdas) * 2 * len(SELECTION_SEEDS) * len(STUDENT_SEEDS)
    payload = {
        "status": "complete" if found == expected and not errors else "incomplete",
        "experiment": "imagenette_entropy_selection_v1",
        "lambdas": lambdas,
        "expected_results": expected,
        "found_results": found,
        "primary_metric": "Final Top-1 at update 4000",
        "interaction": "I(lambda)=[Soft(lambda)-Hard(lambda)]-[Soft(0)-Hard(0)]",
        "errors": errors,
        "summary": summary,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "complete":
        raise RuntimeError(f"entropy selection matrix incomplete: {len(errors)} errors")


if __name__ == "__main__":
    main()
