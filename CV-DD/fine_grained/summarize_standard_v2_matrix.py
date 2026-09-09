"""Summarize the 243-run fine-grained SRe2L++ standard-v2 matrix."""

import argparse
import json
import statistics
from pathlib import Path

from audit_result import audit_payload


DATASETS = {
    "CUB_imsize224": (200, 5794),
    "A_imsize224": (100, 3333),
    "SC_imsize224": (196, 8041),
}
SEEDS = (42, 43, 44)
IPCS = (1, 3, 5)


def stats(values):
    return {
        "mean": statistics.mean(values) if values else None,
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "values": values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", required=True, type=Path)
    args = parser.parse_args()
    rows, missing, invalid = [], [], []
    for dataset, (classes, validation_images) in DATASETS.items():
        for teacher_seed in SEEDS:
            for recovery_seed in SEEDS:
                for ipc in IPCS:
                    for student_seed in SEEDS:
                        path = (args.v2_root / "results" / f"tseed{teacher_seed}" /
                                dataset / f"rseed{recovery_seed}" /
                                f"ipc{ipc}_sseed{student_seed}.json")
                        if not path.is_file():
                            missing.append(str(path.resolve()))
                            continue
                        try:
                            payload = json.loads(path.read_text(encoding="utf-8"))
                            best = audit_payload(payload, classes, validation_images)
                            protocol = payload["standard_protocol"]
                            observed = tuple(protocol[key] for key in (
                                "dataset", "teacher_seed", "recovery_seed", "ipc", "student_seed"
                            ))
                            expected = (dataset, teacher_seed, recovery_seed, ipc, student_seed)
                            if protocol["version"] != "v2" or observed != expected:
                                raise RuntimeError(f"protocol/path mismatch: {observed} != {expected}")
                            rows.append({
                                "dataset": dataset, "teacher_seed": teacher_seed,
                                "recovery_seed": recovery_seed, "ipc": ipc,
                                "student_seed": student_seed, "best_top1": best,
                                "final_top1": payload.get("final_epoch_top1"),
                                "result": str(path.resolve()),
                            })
                        except Exception as error:
                            invalid.append({"result": str(path.resolve()), "error": str(error)})
    groups = []
    for dataset in DATASETS:
        for ipc in IPCS:
            selected = [row for row in rows if row["dataset"] == dataset and row["ipc"] == ipc]
            groups.append({
                "dataset": dataset, "ipc": ipc, "completed": len(selected), "expected": 27,
                "best_top1": stats([row["best_top1"] for row in selected]),
                "final_top1": stats([row["final_top1"] for row in selected]),
            })
    payload = {
        "status": "complete" if len(rows) == 243 and not invalid else "incomplete",
        "protocol_name": "fine_grained_sre2l_standard_protocol",
        "protocol_version": "v2", "expected_results": 243,
        "completed_results": len(rows), "missing_results": len(missing),
        "invalid_results": len(invalid), "groups": groups, "rows": rows,
        "missing": missing, "invalid": invalid,
    }
    output = args.v2_root / "summary/standard_v2_matrix.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({
        "status": payload["status"], "completed": len(rows),
        "expected": 243, "invalid": len(invalid), "output": str(output.resolve()),
    }))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
