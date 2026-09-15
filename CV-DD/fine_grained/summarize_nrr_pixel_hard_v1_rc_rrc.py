"""Summarize NRR pixel images under Aircraft Hard-v1 mild-RC and RRC."""

import argparse
import json
import statistics
from pathlib import Path


MODES = ("mild_rc", "rrc")
ARMS = ("r0", "plain", "cam_protected", "random_protected")
SEEDS = (42, 43, 44)
METRICS = ("best_top1_diagnostic", "final_top1")


def stats(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "values": values}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--r0-root", required=True, type=Path)
    a = p.parse_args()
    rows = {mode: {} for mode in MODES}
    for mode in MODES:
        rows[mode]["r0"] = [json.loads((a.r0_root / f"results/{mode}/r0/sseed{seed}.json").read_text()) for seed in SEEDS]
        for arm in ARMS[1:]:
            rows[mode][arm] = [json.loads((a.root / f"results/{mode}/{arm}/sseed{seed}.json").read_text()) for seed in SEEDS]
    for mode in MODES:
        for arm in ARMS:
            for seed, row in zip(SEEDS, rows[mode][arm]):
                if row.get("status") != "complete" or row.get("student_seed") != seed:
                    raise RuntimeError(f"invalid result {mode}/{arm}/sseed{seed}")
                if row.get("protocol") != f"hard_label_v1_{mode}" or row.get("train_crop_mode") != mode:
                    raise RuntimeError(f"protocol mismatch {mode}/{arm}/sseed{seed}")
                if row.get("updates_completed") != 3000:
                    raise RuntimeError(f"update mismatch {mode}/{arm}/sseed{seed}")
    for mode in MODES:
        for seed_index in range(3):
            hashes = {(rows[mode][arm][seed_index]["imagenet_initial_state_sha256"],
                       rows[mode][arm][seed_index]["initial_head_sha256"]) for arm in ARMS}
            if len(hashes) != 1:
                raise RuntimeError(f"initial state mismatch {mode}/sseed{SEEDS[seed_index]}")
    result = {
        "status": "complete", "protocol": "nrr_pixel_optimization_hard_v1_rc_rrc_v1",
        "student_seeds": list(SEEDS),
        "groups": {mode: {arm: {metric: stats([row[metric] for row in rows[mode][arm]]) for metric in METRICS}
                          for arm in ARMS} for mode in MODES},
        "paired_deltas": {}, "new_student_runs": 18, "new_fkd_sets": 0, "teacher_queries": 0,
    }
    for mode in MODES:
        result["paired_deltas"][mode] = {}
        for left, right in (("plain", "r0"), ("cam_protected", "r0"), ("random_protected", "r0"),
                            ("cam_protected", "plain"), ("cam_protected", "random_protected")):
            result["paired_deltas"][mode][f"{left} minus {right}"] = {
                metric: stats([x[metric] - y[metric] for x, y in zip(rows[mode][left], rows[mode][right])])
                for metric in METRICS
            }
    result["paired_rrc_minus_mild_rc"] = {
        arm: {metric: stats([right[metric] - left[metric]
                             for left, right in zip(rows["mild_rc"][arm], rows["rrc"][arm])])
              for metric in METRICS} for arm in ARMS
    }
    output = a.root / "summary/summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
