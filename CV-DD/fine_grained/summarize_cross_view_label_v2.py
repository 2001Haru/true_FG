"""Compare A/A, B/B and B/A-label Aircraft IPC5 results."""

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
    parser.add_argument("--a-root", required=True, type=Path)
    parser.add_argument("--b-root", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    errors = []
    for student in (42, 43, 44):
        paths = {
            "aa": args.a_root / f"results/entropy_t20/ipc5_sseed{student}.json",
            "bb": args.b_root / f"results/downsample_up/ipc5_sseed{student}.json",
            "ba": args.root / f"results/ipc5_sseed{student}.json",
        }
        try:
            x = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
            for payload in x.values():
                audit_payload(payload, 100, 3333)
            row = {"student_seed": student}
            for name, payload in x.items():
                row[f"{name}_best"] = payload["best_top1"]
                row[f"{name}_final"] = payload["final_epoch_top1"]
                row[f"{name}_result"] = str(paths[name].resolve())
            row["label_rescue_best"] = row["ba_best"] - row["bb_best"]
            row["label_rescue_final"] = row["ba_final"] - row["bb_final"]
            row["remaining_gap_best"] = row["ba_best"] - row["aa_best"]
            row["remaining_gap_final"] = row["ba_final"] - row["aa_final"]
            rows.append(row)
        except Exception as error:
            errors.append({"student_seed": student, "error": str(error)})
    summary = {"count": len(rows)}
    if rows:
        fields = [key for key in rows[0] if key.endswith("_best") or key.endswith("_final")]
        summary.update({field: stats(row[field] for row in rows) for field in fields})
    payload = {
        "status": "complete" if len(rows) == 3 and not errors else "failed",
        "dataset": "A_imsize224", "ipc": 5, "teacher_seed": 42,
        "student_seeds": [42, 43, 44], "rrc_scale": [0.08, 1.0],
        "conditions": {"aa": "A image/A label", "bb": "B image/B label", "ba": "B image/A label"},
        "summary": summary, "rows": rows, "errors": errors,
    }
    output = args.root / "summary/cross_view_label_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": payload["status"], "rows": len(rows), "errors": len(errors), "output": str(output.resolve())}))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
