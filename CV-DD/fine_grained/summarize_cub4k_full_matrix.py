import argparse
import json
import statistics
from pathlib import Path

from audit_result import audit_payload


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser("Summarize full crossed-seed CUB 4k matrix")
    parser.add_argument("--matrix-root", required=True, type=Path)
    parser.add_argument("--standard-root", required=True, type=Path)
    args = parser.parse_args()
    matrix = args.matrix_root.resolve()
    standard = args.standard_root.resolve()
    rows = []
    missing = []
    invalid = []
    for teacher_seed in (42, 43, 44):
        for recovery_seed in (42, 43, 44):
            for ipc in (1, 3, 5):
                for student_seed in (42, 43, 44):
                    path = (matrix / "results" / f"tseed{teacher_seed}" /
                            "CUB_imsize224" / f"rseed{recovery_seed}" /
                            f"ipc{ipc}_sseed{student_seed}.json")
                    baseline_path = (standard / "results" / f"tseed{teacher_seed}" /
                                     "CUB_imsize224" / f"rseed{recovery_seed}" /
                                     f"ipc{ipc}_sseed{student_seed}.json")
                    if not path.is_file():
                        missing.append(str(path))
                        continue
                    try:
                        payload = load(path)
                        baseline = load(baseline_path)
                        best = audit_payload(payload, 200, 5794)
                        baseline_best = audit_payload(baseline, 200, 5794)
                        protocol = payload["cub4k_protocol"]
                        observed = (protocol["teacher_seed"], protocol["recovery_seed"],
                                    protocol["ipc"], protocol["student_seed"])
                        expected = (teacher_seed, recovery_seed, ipc, student_seed)
                        if observed != expected or protocol["recovery_iterations"] != 4000:
                            raise RuntimeError(f"protocol/path mismatch {observed} != {expected}")
                        rows.append({
                            "teacher_seed": teacher_seed,
                            "recovery_seed": recovery_seed,
                            "ipc": ipc,
                            "student_seed": student_seed,
                            "best_top1": best,
                            "final_epoch_top1": payload["final_epoch_top1"],
                            "baseline_10k_best_top1": baseline_best,
                            "delta_4k_minus_10k": best - baseline_best,
                            "result": str(path.resolve()),
                            "baseline_result": str(baseline_path.resolve()),
                            "reused": protocol.get("reused_source") is not None,
                        })
                    except Exception as error:
                        invalid.append({"result": str(path), "error": str(error)})
    groups = []
    for ipc in (1, 3, 5):
        group = [row for row in rows if row["ipc"] == ipc]
        values = [row["best_top1"] for row in group]
        final = [row["final_epoch_top1"] for row in group]
        delta = [row["delta_4k_minus_10k"] for row in group]
        groups.append({
            "ipc": ipc,
            "completed": len(group),
            "expected": 27,
            "best_mean": statistics.mean(values) if values else None,
            "best_sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "final_mean": statistics.mean(final) if final else None,
            "final_sample_std": statistics.stdev(final) if len(final) > 1 else None,
            "paired_delta_mean_4k_minus_10k": statistics.mean(delta) if delta else None,
            "paired_delta_sample_std": statistics.stdev(delta) if len(delta) > 1 else None,
            "paired_delta_minimum": min(delta) if delta else None,
            "paired_delta_maximum": max(delta) if delta else None,
        })
    payload = {
        "status": "complete" if len(rows) == 81 and not invalid else "incomplete",
        "experiment": "cub_recovery_4000_full_crossed_seed_matrix",
        "teacher_seeds": [42, 43, 44],
        "recovery_seeds": [42, 43, 44],
        "student_seeds": [42, 43, 44],
        "ipcs": [1, 3, 5],
        "recovery_iterations": 4000,
        "paired_baseline_iterations": 10000,
        "expected_results": 81,
        "completed_results": len(rows),
        "reused_results": sum(row["reused"] for row in rows),
        "groups": groups,
        "rows": rows,
        "missing": missing,
        "invalid": invalid,
    }
    output = matrix / "summary/cub4k_full_matrix.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "completed_results": len(rows),
        "expected_results": 81, "output": str(output.resolve())
    }, sort_keys=True))


if __name__ == "__main__":
    main()
