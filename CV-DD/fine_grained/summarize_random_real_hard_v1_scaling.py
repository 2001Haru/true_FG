"""Summarize nested Aircraft RandomReal IPC5/10/20 under Hard-v1 mild RC."""

import argparse
import json
import statistics
from pathlib import Path


IPCS = (5, 10, 20)
RSEEDS = (0, 1, 2)
SSEEDS = (42, 43, 44)
METRICS = ("best_top1_diagnostic", "final_top1")


def stats(values):
    values = list(map(float, values))
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "values": values}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    a = p.parse_args()
    rows = {}
    for ipc in IPCS:
        for rseed in RSEEDS:
            for sseed in SSEEDS:
                path = a.root / f"hard_v1/results/rseed{rseed}/ipc{ipc}_sseed{sseed}.json"
                row = json.loads(path.read_text())
                if row.get("status") != "complete" or row.get("ipc") != ipc or row.get("student_seed") != sseed:
                    raise RuntimeError(f"identity mismatch: {path}")
                if row.get("protocol") != "hard_label_v1_mild_rc" or row.get("train_crop_mode") != "mild_rc":
                    raise RuntimeError(f"protocol mismatch: {path}")
                if row.get("updates_completed") != 3000 or row.get("physical_batch_size") != 64:
                    raise RuntimeError(f"training budget mismatch: {path}")
                rows[(ipc, rseed, sseed)] = row
    groups = {}
    for ipc in IPCS:
        group = [rows[(ipc, rseed, sseed)] for rseed in RSEEDS for sseed in SSEEDS]
        groups[str(ipc)] = {
            metric: stats(row[metric] for row in group) for metric in METRICS
        }
        groups[str(ipc)]["by_selection_seed"] = {
            str(rseed): {metric: stats(rows[(ipc, rseed, sseed)][metric] for sseed in SSEEDS) for metric in METRICS}
            for rseed in RSEEDS
        }
    deltas = {}
    for smaller, larger in ((5, 10), (10, 20), (5, 20)):
        deltas[f"ipc{larger} minus ipc{smaller}"] = {
            metric: stats(rows[(larger, rseed, sseed)][metric] - rows[(smaller, rseed, sseed)][metric]
                          for rseed in RSEEDS for sseed in SSEEDS) for metric in METRICS
        }
    result = {"status": "complete", "protocol": "random_real_hard_v1_mild_rc_scaling_v1",
              "dataset": "A_imsize224", "ipcs": list(IPCS), "selection_seeds": list(RSEEDS),
              "student_seeds": list(SSEEDS), "groups": groups, "paired_scaling_deltas": deltas,
              "expected_results": 27, "completed_results": len(rows), "fkd_sets": 0, "teacher_queries": 0}
    output = a.root / "summary/hard_v1_mild_rc.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
