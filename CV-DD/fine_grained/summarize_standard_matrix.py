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


def main() -> None:
    parser = argparse.ArgumentParser("Summarize the standard-protocol crossed-seed matrix")
    parser.add_argument("--standard-root", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    missing = []
    invalid = []
    for dataset, (classes, validation_images) in DATASETS.items():
        for teacher_seed in SEEDS:
            for recovery_seed in SEEDS:
                for ipc in IPCS:
                    for student_seed in SEEDS:
                        path = (args.standard_root / "results" / f"tseed{teacher_seed}" /
                                dataset / f"rseed{recovery_seed}" /
                                f"ipc{ipc}_sseed{student_seed}.json")
                        if not path.is_file():
                            missing.append(str(path.resolve()))
                            continue
                        try:
                            payload = json.loads(path.read_text(encoding="utf-8"))
                            best = audit_payload(payload, classes, validation_images)
                            protocol = payload["standard_protocol"]
                            expected = (dataset, teacher_seed, recovery_seed, ipc, student_seed)
                            observed = (
                                protocol["dataset"], protocol["teacher_seed"],
                                protocol["recovery_seed"], protocol["ipc"],
                                protocol["student_seed"],
                            )
                            if observed != expected:
                                raise RuntimeError(f"seed/path mismatch: {observed} != {expected}")
                            rows.append({
                                "dataset": dataset,
                                "teacher_seed": teacher_seed,
                                "recovery_seed": recovery_seed,
                                "ipc": ipc,
                                "student_seed": student_seed,
                                "best_top1": best,
                                "final_epoch_top1": payload.get("final_epoch_top1"),
                                "result": str(path.resolve()),
                            })
                        except Exception as error:  # keep the full inventory auditable
                            invalid.append({"result": str(path.resolve()), "error": str(error)})

    groups = []
    for dataset in DATASETS:
        for ipc in IPCS:
            values = [row["best_top1"] for row in rows
                      if row["dataset"] == dataset and row["ipc"] == ipc]
            groups.append({
                "dataset": dataset,
                "ipc": ipc,
                "completed": len(values),
                "expected": 27,
                "mean_best_top1": statistics.mean(values) if values else None,
                "sample_std_best_top1": statistics.stdev(values) if len(values) > 1 else None,
                "minimum_best_top1": min(values) if values else None,
                "maximum_best_top1": max(values) if values else None,
            })
    result = {
        "status": "complete" if len(rows) == 243 and not invalid else "incomplete",
        "protocol_name": "standard_protocol",
        "protocol_version": "v1",
        "expected_results": 243,
        "completed_results": len(rows),
        "missing_results": len(missing),
        "invalid_results": len(invalid),
        "groups": groups,
        "rows": rows,
        "missing": missing,
        "invalid": invalid,
    }
    output = args.standard_root / "summary" / "standard_matrix.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "completed_results": len(rows),
        "expected_results": 243,
        "output": str(output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
