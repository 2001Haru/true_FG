"""Summarize the 3 generation seeds x 3 Students x 3 IPC RDED matrix."""

import argparse
import json
import os
import statistics
from pathlib import Path

from audit_result import audit_payload


def stats(values):
    values = list(map(float, values))
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "values": values}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    errors = []
    for ipc in (1, 3, 5):
        for generation in (42, 43, 44):
            for student in (42, 43, 44):
                path = args.root / "results/A_imsize224" / f"gseed{generation}" / f"ipc{ipc}_sseed{student}.json"
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    audit_payload(payload, 100, 3333)
                    if payload["standard_protocol"]["version"] != "v2":
                        raise RuntimeError("protocol mismatch")
                    if payload["rded_original_v2"]["teacher_seed"] != 42:
                        raise RuntimeError("Teacher mismatch")
                    rows.append(
                        {
                            "ipc": ipc,
                            "generation_seed": generation,
                            "student_seed": student,
                            "best_top1": payload["best_top1"],
                            "final_top1": payload["final_epoch_top1"],
                            "result": str(path.resolve()),
                        }
                    )
                except Exception as error:
                    errors.append({"result": str(path.resolve()), "error": str(error)})
    groups = []
    for ipc in (1, 3, 5):
        selected = [row for row in rows if row["ipc"] == ipc]
        group = {"ipc": ipc, "count": len(selected)}
        if selected:
            group.update(
                {
                    "best_top1": stats(row["best_top1"] for row in selected),
                    "final_top1": stats(row["final_top1"] for row in selected),
                    "generation_means": {
                        str(seed): statistics.mean(
                            row["best_top1"] for row in selected if row["generation_seed"] == seed
                        )
                        for seed in (42, 43, 44)
                    },
                }
            )
        groups.append(group)
    payload = {
        "status": "complete" if len(rows) == 27 and not errors else "failed",
        "expected": 27,
        "completed": len(rows),
        "dataset": "A_imsize224",
        "teacher_seed": 42,
        "generation_seeds": [42, 43, 44],
        "student_seeds": [42, 43, 44],
        "ipcs": [1, 3, 5],
        "generation": "released RDED joint Top-(IPC*4) factor-2 mosaic",
        "supervision": "fixed SRe2L++ Teacher42 FKD soft labels",
        "student_protocol": "standard_protocol_v2",
        "groups": groups,
        "rows": rows,
        "errors": errors,
    }
    output = args.root / "summary/rded_original_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": payload["status"], "completed": len(rows), "errors": len(errors), "output": str(output.resolve())}))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
