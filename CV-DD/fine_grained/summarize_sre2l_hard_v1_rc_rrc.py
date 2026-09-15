"""Summarize Aircraft IPC3 SRe2L++ Hard-v1 mild-RC versus RRC."""

import argparse
import json
import statistics
from pathlib import Path


SEEDS = (42, 43, 44)
MODES = ("mild_rc", "rrc")
METRICS = ("best_top1_diagnostic", "final_top1")


def stats(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "values": values}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--soft-reference-root", required=True, type=Path)
    a = p.parse_args()
    rows = {
        mode: [json.loads((a.root / f"results/{mode}/sseed{seed}.json").read_text()) for seed in SEEDS]
        for mode in MODES
    }
    for mode in MODES:
        for seed, row in zip(SEEDS, rows[mode]):
            if row["status"] != "complete" or row["student_seed"] != seed:
                raise RuntimeError(f"invalid {mode}/sseed{seed} result")
            if row["protocol"] != f"hard_label_v1_{mode}" or row["train_crop_mode"] != mode:
                raise RuntimeError(f"protocol mismatch: {mode}/sseed{seed}")
            if row["updates_completed"] != 3000 or row["total_optimizer_updates"] != 3000:
                raise RuntimeError(f"update mismatch: {mode}/sseed{seed}")
    for index in range(len(SEEDS)):
        hashes = {(rows[mode][index]["imagenet_initial_state_sha256"], rows[mode][index]["initial_head_sha256"])
                  for mode in MODES}
        if len(hashes) != 1:
            raise RuntimeError(f"initial state mismatch for sseed{SEEDS[index]}")
    soft = [json.loads((a.soft_reference_root / f"ipc3_sseed{seed}.json").read_text()) for seed in SEEDS]
    result = {
        "status": "complete", "protocol": "sre2l_aircraft_ipc3_hard_v1_rc_rrc_v1",
        "teacher_seed": 42, "recovery_seed": 42, "student_seeds": list(SEEDS),
        "groups": {mode: {metric: stats([row[metric] for row in rows[mode]]) for metric in METRICS} for mode in MODES},
        "paired_rrc_minus_mild_rc": {
            metric: stats([right[metric] - left[metric] for left, right in zip(rows["mild_rc"], rows["rrc"])])
            for metric in METRICS
        },
        "soft_v2_reference": {
            "best_top1": stats([row["best_top1"] for row in soft]),
            "final_epoch_top1": stats([row["final_epoch_top1"] for row in soft]),
            "note": "same Teacher42/Recovery42/IPC3/Student seeds, but full Soft-v2 protocol differs in labels, optimizer, batch and augmentation",
        },
        "new_student_runs": 6, "new_fkd_sets": 0, "teacher_queries": 0,
    }
    output = a.root / "summary/summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
