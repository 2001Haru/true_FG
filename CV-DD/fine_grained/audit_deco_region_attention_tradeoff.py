"""Audit attention-score tradeoffs among current, random, and coverage regions."""

import argparse
import json
import statistics
from pathlib import Path


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def describe(values):
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
        "min": min(values),
        "p25": quantile(values, 0.25),
        "median": quantile(values, 0.5),
        "p75": quantile(values, 0.75),
        "max": max(values),
        "values_count": len(values),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--construction-manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    manifest = json.loads(a.construction_manifest.read_text())
    if manifest.get("status") != "complete" or len(manifest.get("regions_detail", [])) != 1200:
        raise RuntimeError("expected completed 1200-source construction manifest")
    methods = {
        "current_fg": lambda row: 0,
        "candidate_random": lambda row: int(row["candidate_random_candidate_index"]),
        "joint_coverage": lambda row: int(row["joint_coverage_candidate_index"]),
    }
    rows = manifest["regions_detail"]
    selected_indices, selected_attention = {}, {}
    groups = {}
    for method, chooser in methods.items():
        indices = [chooser(row) for row in rows]
        attention = [float(row["candidates"][index]["attention_mean"]) for row, index in zip(rows, indices)]
        relative = [score / float(row["candidates"][0]["attention_mean"])
                    for row, index, score in zip(rows, indices, attention)]
        delta = [score - float(row["candidates"][0]["attention_mean"])
                 for row, score in zip(rows, attention)]
        selected_indices[method] = indices
        selected_attention[method] = attention
        groups[method] = {
            "candidate_index_histogram": {
                str(index): indices.count(index)
                for index in range(int(manifest["candidates_per_source"]))
            },
            "candidate_index_mean": statistics.mean(indices),
            "attention": describe(attention),
            "attention_ratio_to_current_per_source": describe(relative),
            "attention_delta_from_current_per_source": describe(delta),
        }
    pairwise = {}
    for left, right in (("joint_coverage", "candidate_random"),
                        ("candidate_random", "current_fg"),
                        ("joint_coverage", "current_fg")):
        differences = [x - y for x, y in zip(selected_attention[left], selected_attention[right])]
        tolerance = 1e-12
        pairwise[f"{left} minus {right}"] = {
            "attention_difference": describe(differences),
            "left_lower_fraction": sum(value < -tolerance for value in differences) / len(differences),
            "equal_fraction": sum(abs(value) <= tolerance for value in differences) / len(differences),
            "left_higher_fraction": sum(value > tolerance for value in differences) / len(differences),
            "same_candidate_fraction": sum(x == y for x, y in zip(selected_indices[left], selected_indices[right])) / len(rows),
        }
    result = {
        "status": "complete",
        "protocol": "deco_region_attention_tradeoff_audit_v1",
        "construction_manifest": str(a.construction_manifest.resolve()),
        "attention_definition": "mean TransFG attention-rollout response within the 119x119 candidate window on the 224x224 reference",
        "groups": groups,
        "pairwise": pairwise,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
