"""Summarize new RRC 0.5-1 runs against existing RRC 0.08-1 references."""

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
    parser.add_argument("--random-reference-root", required=True, type=Path)
    parser.add_argument("--rded-reference-root", required=True, type=Path)
    args = parser.parse_args()
    definitions = {
        "random_real": {
            "new": args.root / "results/random_real",
            "old": args.random_reference_root / "results/A_imsize224/rseed0",
        },
        "rded": {
            "new": args.root / "results/rded",
            "old": args.rded_reference_root / "results/A_imsize224/gseed42",
        },
    }
    groups = []
    rows = []
    errors = []
    for source, paths in definitions.items():
        for student in (42, 43, 44):
            new_path = paths["new"] / f"ipc3_sseed{student}.json"
            old_path = paths["old"] / f"ipc3_sseed{student}.json"
            try:
                new = json.loads(new_path.read_text(encoding="utf-8"))
                old = json.loads(old_path.read_text(encoding="utf-8"))
                audit_payload(new, 100, 3333)
                audit_payload(old, 100, 3333)
                if new["rrc_sensitivity_v2"]["source_kind"] != source:
                    raise RuntimeError("source mismatch")
                rows.append(
                    {
                        "source": source,
                        "student_seed": student,
                        "reference_best": old["best_top1"],
                        "new_best": new["best_top1"],
                        "best_delta": new["best_top1"] - old["best_top1"],
                        "reference_final": old["final_epoch_top1"],
                        "new_final": new["final_epoch_top1"],
                        "final_delta": new["final_epoch_top1"] - old["final_epoch_top1"],
                        "new_result": str(new_path.resolve()),
                        "reference_result": str(old_path.resolve()),
                    }
                )
            except Exception as error:
                errors.append({"source": source, "student_seed": student, "error": str(error)})
    for source in definitions:
        selected = [row for row in rows if row["source"] == source]
        group = {"source": source, "count": len(selected)}
        if selected:
            for field in ("reference_best", "new_best", "best_delta", "reference_final", "new_final", "final_delta"):
                group[field] = stats(row[field] for row in selected)
        groups.append(group)
    payload = {
        "status": "complete" if len(rows) == 6 and not errors else "failed",
        "dataset": "A_imsize224",
        "ipc": 3,
        "teacher_seed": 42,
        "random_real_selection_seed": 0,
        "rded_generation_seed": 42,
        "student_seeds": [42, 43, 44],
        "reference_rrc_scale": [0.08, 1.0],
        "new_rrc_scale": [0.5, 1.0],
        "groups": groups,
        "rows": rows,
        "errors": errors,
    }
    output = args.root / "summary/rrc_sensitivity_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": payload["status"], "rows": len(rows), "errors": len(errors), "output": str(output.resolve())}))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
