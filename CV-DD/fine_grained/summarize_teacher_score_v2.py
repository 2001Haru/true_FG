"""Summarize Aircraft Teacher-score selection under standard protocol v2."""

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
    args = parser.parse_args()
    selection = json.loads((args.root / "selection/selection_manifest.json").read_text(encoding="utf-8"))
    rows = []
    errors = []
    expected = [("entropy_t20", ipc, seed) for ipc in (1, 3, 5) for seed in (42, 43, 44)]
    ce_differs = selection["overlaps"]["entropy_t20__vs__true_ce_t1"]["3"]["intersection"] < 300
    if ce_differs:
        expected.extend(("true_ce_t1", 3, seed) for seed in (42, 43, 44))
    for method, ipc, student in expected:
        path = args.root / f"results/{method}/ipc{ipc}_sseed{student}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            audit_payload(payload, 100, 3333)
            metadata = payload["teacher_score_selection_v2"]
            if metadata["selection_method"] != method or metadata["ipc"] != ipc:
                raise RuntimeError("selection metadata mismatch")
            rows.append(
                {"method": method, "ipc": ipc, "student_seed": student,
                 "best_top1": payload["best_top1"], "final_top1": payload["final_epoch_top1"],
                 "result": str(path.resolve())}
            )
        except Exception as error:
            errors.append({"method": method, "ipc": ipc, "student_seed": student, "error": str(error)})
    groups = []
    for method, ipc in sorted({(row["method"], row["ipc"]) for row in rows}):
        selected = [row for row in rows if row["method"] == method and row["ipc"] == ipc]
        groups.append({"method": method, "ipc": ipc, "count": len(selected),
                       "best_top1": stats(row["best_top1"] for row in selected),
                       "final_top1": stats(row["final_top1"] for row in selected)})
    random = json.loads((args.random_reference_root / "summary/random_real_soft_v2.json").read_text(encoding="utf-8"))
    comparisons = []
    for ipc in (1, 3, 5):
        entropy_group = next(group for group in groups if group["method"] == "entropy_t20" and group["ipc"] == ipc)
        random_group = next(group for group in random["groups"] if group["ipc"] == ipc)
        comparisons.append({
            "ipc": ipc,
            "entropy_t20_minus_random_real_best": entropy_group["best_top1"]["mean"] - random_group["best_top1"]["mean"],
            "entropy_t20_minus_random_real_final": entropy_group["final_top1"]["mean"] - random_group["final_top1"]["mean"],
            "random_real_best": random_group["best_top1"],
            "random_real_final": random_group["final_top1"],
        })
    ce_comparison = None
    if ce_differs and not errors:
        entropy_rows = {row["student_seed"]: row for row in rows if row["method"] == "entropy_t20" and row["ipc"] == 3}
        ce_rows = {row["student_seed"]: row for row in rows if row["method"] == "true_ce_t1" and row["ipc"] == 3}
        ce_comparison = {
            "ce_minus_entropy_best": stats(ce_rows[seed]["best_top1"] - entropy_rows[seed]["best_top1"] for seed in (42, 43, 44)),
            "ce_minus_entropy_final": stats(ce_rows[seed]["final_top1"] - entropy_rows[seed]["final_top1"] for seed in (42, 43, 44)),
        }
    fkd_entropy = {}
    for method, ipc, _ in expected:
        key = f"{method}_ipc{ipc}"
        if key not in fkd_entropy:
            path = args.root / f"fkd/{method}/ipc{ipc}_bs20_ipc{ipc}/t20_entropy_audit.json"
            if path.is_file():
                audit = json.loads(path.read_text(encoding="utf-8"))
                fkd_entropy[key] = {field: audit[field] for field in ("views", "mean", "sample_std", "normalized_mean", "quantiles")}
    payload = {
        "status": "complete" if len(rows) == len(expected) and not errors else "failed",
        "dataset": "A_imsize224", "teacher_seed": 42, "student_seeds": [42, 43, 44],
        "primary_method": "entropy_t20", "ipcs": [1, 3, 5],
        "ce_ipc3_triggered": ce_differs, "expected_results": len(expected), "completed": len(rows),
        "selection_overlaps": selection["overlaps"],
        "selection_summaries": selection["selection_summaries"],
        "groups": groups, "comparisons_to_random_real": comparisons,
        "ce_ipc3_comparison": ce_comparison, "fkd_t20_entropy": fkd_entropy,
        "rows": rows, "errors": errors,
    }
    output = args.root / "summary/teacher_score_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": payload["status"], "completed": len(rows), "expected": len(expected), "errors": len(errors), "output": str(output.resolve())}))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
