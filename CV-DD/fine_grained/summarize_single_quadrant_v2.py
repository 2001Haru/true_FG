"""Compare RDED single-quadrant views with whole-mosaic RRC 0.5-1."""

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
    parser.add_argument("--whole-mosaic-root", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    errors = []
    for student in (42, 43, 44):
        new_path = args.root / f"results/ipc3_sseed{student}.json"
        reference_path = args.whole_mosaic_root / f"results/rded/ipc3_sseed{student}.json"
        try:
            new = json.loads(new_path.read_text(encoding="utf-8"))
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            audit_payload(new, 100, 3333)
            audit_payload(reference, 100, 3333)
            rows.append(
                {
                    "student_seed": student,
                    "whole_best": reference["best_top1"],
                    "quadrant_best": new["best_top1"],
                    "best_delta": new["best_top1"] - reference["best_top1"],
                    "whole_final": reference["final_epoch_top1"],
                    "quadrant_final": new["final_epoch_top1"],
                    "final_delta": new["final_epoch_top1"] - reference["final_epoch_top1"],
                    "result": str(new_path.resolve()),
                    "reference": str(reference_path.resolve()),
                }
            )
        except Exception as error:
            errors.append({"student_seed": student, "error": str(error)})
    summary = {"count": len(rows)}
    if rows:
        for field in ("whole_best", "quadrant_best", "best_delta", "whole_final", "quadrant_final", "final_delta"):
            summary[field] = stats(row[field] for row in rows)
    payload = {
        "status": "complete" if len(rows) == 3 and not errors else "failed",
        "dataset": "A_imsize224",
        "ipc": 3,
        "teacher_seed": 42,
        "generation_seed": 42,
        "student_seeds": [42, 43, 44],
        "rrc_scale": [0.5, 1.0],
        "comparison": "single quadrant minus whole 2x2 mosaic",
        "summary": summary,
        "rows": rows,
        "errors": errors,
    }
    output = args.root / "summary/single_quadrant_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": payload["status"], "rows": len(rows), "errors": len(errors), "output": str(output.resolve())}))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
