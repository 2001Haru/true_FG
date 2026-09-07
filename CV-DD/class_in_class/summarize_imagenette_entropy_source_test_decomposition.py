"""Derive exact high-entropy-tail versus remaining-test identity values."""

import argparse
import json
import math
import statistics
from pathlib import Path

from scipy.stats import t

from imagenette_entropy_protocol import atomic_json, file_sha256


def stats(values):
    values = list(map(float, values))
    mean = statistics.mean(values)
    sample_std = statistics.stdev(values)
    standard_error = sample_std / math.sqrt(len(values))
    critical = float(t.ppf(0.975, len(values) - 1))
    t_value = mean / standard_error if standard_error else float("inf")
    return {
        "mean": mean,
        "sample_std": sample_std,
        "values": values,
        "positive": sum(value > 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "negative": sum(value < 0 for value in values),
        "paired_t_95_ci": [mean - critical * standard_error, mean + critical * standard_error],
        "paired_t_two_sided_p": float(2 * t.sf(abs(t_value), len(values) - 1)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    output = {}
    rows = []
    for group in ("high", "low"):
        measures = {
            metric: {key: [] for key in ("tail20", "rest80", "all", "rest_minus_tail")}
            for metric in ("accuracy", "nll")
        }
        for row in source["checkpoint_readout"]["tuples"]:
            if row["group"] != group:
                continue
            total = row["original"]["all"]["images"]
            conflict = row["original"]["concentrated_conflict"]["images"]
            insufficiency = row["original"]["diffuse_insufficiency"]["images"]
            tail = conflict + insufficiency
            rest = total - tail
            derived = {
                "group": group,
                "selection_seed": row["selection_seed"],
                "student_seed": row["student_seed"],
                "total_images": total,
                "conflict_images": conflict,
                "insufficiency_images": insufficiency,
                "tail_images": tail,
                "rest_images": rest,
            }
            for metric in ("accuracy", "nll"):
                # Positive identity value always means original labels are better:
                # higher accuracy or lower NLL than the permuted-label model.
                sign = 1.0 if metric == "accuracy" else -1.0
                def identity_value(stratum):
                    return sign * (
                        row["original"][stratum][metric]
                        - row["permuted"][stratum][metric]
                    )
                conflict_value = identity_value("concentrated_conflict")
                insufficiency_value = identity_value("diffuse_insufficiency")
                tail_value = (
                    conflict_value * conflict + insufficiency_value * insufficiency
                ) / tail
                all_value = identity_value("all")
                rest_value = (all_value * total - tail_value * tail) / rest
                derived[f"{metric}_identity_value_tail20"] = tail_value
                derived[f"{metric}_identity_value_rest80"] = rest_value
                derived[f"{metric}_identity_value_all"] = all_value
                derived[f"{metric}_identity_value_rest_minus_tail"] = rest_value - tail_value
                measures[metric]["tail20"].append(tail_value)
                measures[metric]["rest80"].append(rest_value)
                measures[metric]["all"].append(all_value)
                measures[metric]["rest_minus_tail"].append(rest_value - tail_value)
            rows.append(derived)
        output[group] = {
            metric: {key: stats(values) for key, values in groups.items()}
            for metric, groups in measures.items()
        }
    payload = {
        "status": "complete",
        "source": str(args.source.resolve()),
        "source_sha256": file_sha256(args.source.resolve()),
        "definition": "tail20 is the exact union of per-class high-entropy concentrated-conflict and diffuse-insufficiency strata; rest80 is its complement; values are derived per tuple before summary",
        "image_counts": {"all": 3925, "tail20": 788, "rest80": 3137},
        "image_fractions": {"tail20": 788 / 3925, "rest80": 3137 / 3925},
        "summary": output,
        "tuples": rows,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"status": "complete", "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
