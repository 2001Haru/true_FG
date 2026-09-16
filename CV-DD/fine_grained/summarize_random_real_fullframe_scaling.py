"""Summarize Aircraft RandomReal IPC5/10/20 Soft-v2 fullframe+CutMix scaling."""

import argparse
import json
import statistics
from pathlib import Path

from audit_result import audit_payload


IPCS = (5, 10, 20)
SELECTIONS = (0, 1, 2)
STUDENTS = (42, 43, 44)


def stats(values):
    values = list(map(float, values))
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "values": values}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    a = p.parse_args()
    rows = {}
    for ipc in IPCS:
        for selection in SELECTIONS:
            for student in STUDENTS:
                path = a.root / f"results/rseed{selection}/ipc{ipc}_sseed{student}.json"
                payload = json.loads(path.read_text())
                audit_payload(payload, 100, 3333)
                if payload["student_seed"] != student or payload["standard_protocol"]["ipc"] != ipc:
                    raise RuntimeError(f"result identity mismatch: {path}")
                if payload["standard_protocol"]["version"] != "v2" or payload["standard_protocol"]["selection_seed"] != selection:
                    raise RuntimeError(f"provenance mismatch: {path}")
                rows[(ipc, selection, student)] = payload
    groups = {}
    for ipc in IPCS:
        group = [rows[(ipc, selection, student)] for selection in SELECTIONS for student in STUDENTS]
        groups[str(ipc)] = {
            "best_top1": stats(row["best_top1"] for row in group),
            "final_top1": stats(row["final_epoch_top1"] for row in group),
            "by_selection_seed": {
                str(selection): {
                    "best_top1": stats(rows[(ipc, selection, student)]["best_top1"] for student in STUDENTS),
                    "final_top1": stats(rows[(ipc, selection, student)]["final_epoch_top1"] for student in STUDENTS),
                } for selection in SELECTIONS
            },
        }
    deltas = {}
    for smaller, larger in ((5, 10), (10, 20), (5, 20)):
        deltas[f"ipc{larger} minus ipc{smaller}"] = {
            "best_top1": stats(rows[(larger, selection, student)]["best_top1"] - rows[(smaller, selection, student)]["best_top1"]
                               for selection in SELECTIONS for student in STUDENTS),
            "final_top1": stats(rows[(larger, selection, student)]["final_epoch_top1"] - rows[(smaller, selection, student)]["final_epoch_top1"]
                                for selection in SELECTIONS for student in STUDENTS),
        }
    result = {"status": "complete", "protocol": "random_real_soft_v2_fullframe_cutmix_scaling_v1",
              "dataset": "A_imsize224", "ipcs": list(IPCS), "selection_seeds": list(SELECTIONS),
              "student_seeds": list(STUDENTS), "groups": groups, "paired_scaling_deltas": deltas,
              "expected_results": 27, "completed_results": len(rows), "new_fkd_sets": 9}
    output = a.root / "summary/summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
