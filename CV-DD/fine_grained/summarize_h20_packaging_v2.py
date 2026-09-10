"""Summarize H20 IPC5 original/downsample/mosaic construction arms."""

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
    parser.add_argument("--original-root", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    errors = []
    for student in (42, 43, 44):
        paths = {
            "original": args.original_root / f"results/entropy_t20/ipc5_sseed{student}.json",
            "downsample_up": args.root / f"results/downsample_up/ipc5_sseed{student}.json",
            "same_source_mosaic": args.root / f"results/same_source_mosaic/ipc5_sseed{student}.json",
        }
        try:
            payloads = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
            for payload in payloads.values():
                audit_payload(payload, 100, 3333)
            row = {"student_seed": student}
            for arm, payload in payloads.items():
                row[f"{arm}_best"] = payload["best_top1"]
                row[f"{arm}_final"] = payload["final_epoch_top1"]
                row[f"{arm}_result"] = str(paths[arm].resolve())
            row["downsample_minus_original_best"] = row["downsample_up_best"] - row["original_best"]
            row["downsample_minus_original_final"] = row["downsample_up_final"] - row["original_final"]
            row["mosaic_minus_downsample_best"] = row["same_source_mosaic_best"] - row["downsample_up_best"]
            row["mosaic_minus_downsample_final"] = row["same_source_mosaic_final"] - row["downsample_up_final"]
            row["mosaic_minus_original_best"] = row["same_source_mosaic_best"] - row["original_best"]
            row["mosaic_minus_original_final"] = row["same_source_mosaic_final"] - row["original_final"]
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
        "source_selection": "entropy_t20", "student_seeds": [42, 43, 44],
        "rrc_scale": [0.08, 1.0], "summary": summary, "rows": rows, "errors": errors,
    }
    output = args.root / "summary/h20_packaging_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": payload["status"], "rows": len(rows), "errors": len(errors), "output": str(output.resolve())}))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
