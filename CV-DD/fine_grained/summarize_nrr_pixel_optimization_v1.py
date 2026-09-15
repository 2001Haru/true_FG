"""Summarize the Aircraft IPC3 NRR-inspired pixel construction pilot."""

import argparse
import json
import statistics
from pathlib import Path


ARMS = ("r0", "plain", "cam_protected", "random_protected")
SEEDS = (42, 43, 44)
METRICS = ("best_top1", "final_epoch_top1")


def stats(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "values": values}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--r0-results", required=True, type=Path)
    a = p.parse_args()
    roots = {"r0": a.r0_results, **{arm: a.root / f"results/{arm}" for arm in ARMS[1:]}}
    rows = {arm: [json.loads((root / f"ipc3_sseed{seed}.json").read_text()) for seed in SEEDS]
            for arm, root in roots.items()}
    for arm, group in rows.items():
        for seed, row in zip(SEEDS, group):
            if row.get("student_seed") != seed or row.get("epochs") != 400 or row.get("temperature") != 20:
                raise RuntimeError(f"result protocol mismatch: {arm}/sseed{seed}")
    construction = json.loads((a.root / "construction/construction_manifest.json").read_text())
    if construction.get("status") != "complete":
        raise RuntimeError("construction manifest incomplete")
    result = {
        "status": "complete", "protocol": "aircraft_ipc3_nrr_inspired_pixel_optimization_v1",
        "student_seeds": list(SEEDS),
        "groups": {arm: {metric: stats([row[metric] for row in group]) for metric in METRICS}
                   for arm, group in rows.items()},
        "paired_deltas": {}, "construction_pixel_audit": construction["pixel_audit"],
        "new_fkd_sets": 3, "new_student_runs": 9,
    }
    for left, right in (("plain", "r0"), ("cam_protected", "r0"), ("random_protected", "r0"),
                        ("cam_protected", "plain"), ("cam_protected", "random_protected")):
        result["paired_deltas"][f"{left} minus {right}"] = {
            metric: stats([x[metric] - y[metric] for x, y in zip(rows[left], rows[right])])
            for metric in METRICS
        }
    output = a.root / "summary/summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
