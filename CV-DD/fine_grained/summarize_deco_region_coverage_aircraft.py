"""Summarize Soft-v2 and Hard-v1 RRC DeCO region-coverage controls."""

import argparse
import json
import statistics
from pathlib import Path


METHODS = ("candidate_random", "joint_coverage")
SEEDS = (42, 43, 44)


def stats(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "values": values}


def load_group(root, template):
    return [json.loads((root / template.format(seed=seed)).read_text()) for seed in SEEDS]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--soft-fg-reference", required=True, type=Path)
    p.add_argument("--hard-fg-reference", required=True, type=Path)
    a = p.parse_args()
    construction = json.loads((a.root / "construction/construction_manifest.json").read_text())
    if construction["status"] != "complete" or construction["regions"] != 1200:
        raise RuntimeError("invalid construction manifest")
    rows = {"soft": {}, "hard_v1_rrc": {}}
    for method in METHODS:
        rows["soft"][method] = load_group(a.root, f"results/soft/{method}/ipc3_sseed{{seed}}.json")
        rows["hard_v1_rrc"][method] = load_group(a.root, f"results/hard_v1_rrc/{method}/sseed{{seed}}.json")
    rows["soft"]["current_fg"] = load_group(a.soft_fg_reference, "ipc3_sseed{seed}.json")
    rows["hard_v1_rrc"]["current_fg"] = load_group(a.hard_fg_reference, "sseed{seed}.json")
    metric_names = {
        "soft": ("best_top1", "final_epoch_top1"),
        "hard_v1_rrc": ("best_top1_diagnostic", "final_top1"),
    }
    result = {
        "status": "complete", "protocol": "deco_region_joint_coverage_soft_hard_v1_rrc_v1",
        "construction_manifest": str((a.root / "construction/construction_manifest.json").resolve()),
        "selection_summary": construction["selection_summary"], "groups": {}, "paired_deltas": {},
        "student_seeds": list(SEEDS), "new_fkd_sets": 2, "new_soft_students": 6, "new_hard_students": 6,
    }
    for protocol, groups in rows.items():
        metrics = metric_names[protocol]
        result["groups"][protocol] = {
            method: {metric: stats([row[metric] for row in group]) for metric in metrics}
            for method, group in groups.items()
        }
        result["paired_deltas"][protocol] = {}
        for left, right in (("candidate_random", "current_fg"), ("joint_coverage", "current_fg"), ("joint_coverage", "candidate_random")):
            result["paired_deltas"][protocol][f"{left} minus {right}"] = {
                metric: stats([x[metric] - y[metric] for x, y in zip(groups[left], groups[right])])
                for metric in metrics
            }
        for seed_index, seed in enumerate(SEEDS):
            for method in METHODS:
                row = groups[method][seed_index]
                if row.get("student_seed") != seed:
                    raise RuntimeError(f"student seed mismatch: {protocol}/{method}/{seed}")
            if protocol == "hard_v1_rrc":
                for method in METHODS:
                    row = groups[method][seed_index]
                    if row["status"] != "complete" or row["protocol"] != "hard_label_v1_rrc":
                        raise RuntimeError(f"Hard-v1 protocol mismatch: {method}/{seed}")
    output = a.root / "summary/summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
