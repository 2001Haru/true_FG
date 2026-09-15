"""Summarize Aircraft IPC3 spherical K-means K=3 and Global-center Top3 under Hard-v1."""
import argparse
import json
import statistics
from pathlib import Path


def stats(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "values": values}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--r0-results", required=True, type=Path)
    a = p.parse_args()
    r0 = [json.loads((a.r0_results / f"sseed{s}.json").read_text()) for s in (42, 43, 44)]
    top3 = [json.loads((a.root / f"results/global_center_top3/sseed{s}.json").read_text()) for s in range(42, 48)]
    kmeans = {(rseed, sseed): json.loads((a.root / f"results/spherical_kmeans3_rseed{rseed}/sseed{sseed}.json").read_text())
              for rseed in (0, 1, 2) for sseed in (42, 43, 44)}
    for rows in (r0, top3, list(kmeans.values())):
        for row in rows:
            assert row["status"] == "complete" and row["protocol"] in ("hard_label_v1", "hard_label_v1_mild_rc")
            assert row["updates_completed"] == 3000 and row["physical_batch_size"] == 64
            assert not row["cutmix"] and not row["mixup"]
    metrics = ("best_top1_diagnostic", "final_top1")
    groups = {
        "r0_random_seed0": {m: stats([row[m] for row in r0]) for m in metrics},
        "global_center_top3": {m: stats([row[m] for row in top3]) for m in metrics},
        "spherical_kmeans3": {m: stats([row[m] for row in kmeans.values()]) for m in metrics},
    }
    top3_common = {row["student_seed"]: row for row in top3 if row["student_seed"] in (42, 43, 44)}
    r0_by_student = {row["student_seed"]: row for row in r0}
    kmeans_by_student = {s: {m: statistics.mean(kmeans[r, s][m] for r in (0, 1, 2)) for m in metrics}
                         for s in (42, 43, 44)}
    contrasts = {
        "global_center_top3 minus r0 (common Students42-44)": {
            m: stats([top3_common[s][m] - r0_by_student[s][m] for s in (42, 43, 44)]) for m in metrics},
        "mean_kmeans3_selection minus r0 (paired by Student)": {
            m: stats([kmeans_by_student[s][m] - r0_by_student[s][m] for s in (42, 43, 44)]) for m in metrics},
        "global_center_top3 minus mean_kmeans3_selection (paired by Student)": {
            m: stats([top3_common[s][m] - kmeans_by_student[s][m] for s in (42, 43, 44)]) for m in metrics},
    }
    result = {"status": "complete", "protocol": "dino_ipc3_top3_kmeans3_hard_v1",
              "groups": groups, "paired_contrasts": contrasts,
              "r0_scope": "selection seed0 only", "kmeans_selection_seeds": [0, 1, 2],
              "kmeans_student_seeds": [42, 43, 44], "top3_student_seeds": list(range(42, 48)),
              "new_student_runs": 15}
    out = a.root / "summary/summary.json";out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n");print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
