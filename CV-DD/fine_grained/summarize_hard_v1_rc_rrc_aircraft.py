"""Summarize Aircraft IPC3 Hard-v1 mild-RC versus RRC matrix."""
import argparse
import json
import statistics
from pathlib import Path


MODES = ("mild_rc", "rrc")
METHODS = ("r0", "random_regions", "fg_regions")
SEEDS = (42, 43, 44)
METRICS = ("best_top1_diagnostic", "final_top1")


def stats(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "values": values}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    a = p.parse_args()
    rows = {mode: {method: [json.loads((a.root / f"results/{mode}/{method}/sseed{s}.json").read_text())
                           for s in SEEDS] for method in METHODS} for mode in MODES}
    for mode in MODES:
        for method in METHODS:
            for row, seed in zip(rows[mode][method], SEEDS):
                assert row["status"] == "complete" and row["student_seed"] == seed
                assert row["protocol"] == f"hard_label_v1_{mode}" and row["train_crop_mode"] == mode
                assert row["total_optimizer_updates"] == row["updates_completed"] == 3000
                assert row["physical_batch_size"] == 64 and row["gradient_accumulation_steps"] == 1
                assert not row["cutmix"] and not row["mixup"]
    for seed_index in range(len(SEEDS)):
        hashes = {(rows[mode][method][seed_index]["imagenet_initial_state_sha256"],
                   rows[mode][method][seed_index]["initial_head_sha256"])
                  for mode in MODES for method in METHODS}
        assert len(hashes) == 1
    groups = {mode: {method: {metric: stats([r[metric] for r in rows[mode][method]])
                             for metric in METRICS} for method in METHODS} for mode in MODES}
    contrasts = {}
    for method in METHODS:
        contrasts[f"{method}: rrc minus mild_rc"] = {
            metric: stats([x[metric] - y[metric] for x, y in zip(rows["rrc"][method], rows["mild_rc"][method])])
            for metric in METRICS}
    for mode in MODES:
        for left, right in (("random_regions", "r0"), ("fg_regions", "r0"), ("fg_regions", "random_regions")):
            contrasts[f"{mode}: {left} minus {right}"] = {
                metric: stats([x[metric] - y[metric] for x, y in zip(rows[mode][left], rows[mode][right])])
                for metric in METRICS}
    result = {"status": "complete", "protocol": "hard_label_v1_rc_rrc_aircraft_ipc3_v1",
              "groups": groups, "paired_contrasts": contrasts, "student_seeds": list(SEEDS),
              "new_student_runs": 18, "new_fkd_sets": 0, "teacher_queries": 0}
    out = a.root / "summary/summary.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
