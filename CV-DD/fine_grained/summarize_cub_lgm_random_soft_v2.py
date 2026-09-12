"""Summarize CUB IPC3 LGM and RandomReal standard-v2 Soft results."""

import argparse
import json
import os
import statistics
from pathlib import Path


def stats(values):
    values = list(map(float, values))
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "values": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    rows, errors = [], []
    paths = []
    for student in (42, 43, 44):
        paths.append(("lgm", None, student, args.root / f"results/lgm/ipc3_sseed{student}.json"))
    for selection in (0, 1, 2):
        for student in (42, 43, 44):
            paths.append(("random_real", selection, student,
                          args.root / f"results/random_real/rseed{selection}/ipc3_sseed{student}.json"))
    for method, source_seed, student, path in paths:
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            meta = result["cub_external_soft_v2"]
            if meta["status"] != "complete" or meta["method"] != method:
                raise RuntimeError("invalid result provenance")
            rows.append({"method": method, "source_seed": source_seed, "student_seed": student,
                         "best_top1": result["best_top1"], "final_top1": result["final_epoch_top1"],
                         "best_epoch": result["best_epoch"], "initial_model_sha256": result["initial_model_sha256"],
                         "result": str(path.resolve())})
        except Exception as error:
            errors.append({"path": str(path), "error": str(error)})
    groups = []
    for method in ("lgm", "random_real"):
        selected = [row for row in rows if row["method"] == method]
        groups.append({"method": method, "count": len(selected),
                       "best_top1": stats(row["best_top1"] for row in selected),
                       "final_top1": stats(row["final_top1"] for row in selected),
                       "best_epochs": [row["best_epoch"] for row in selected]})
    hash_audit = {}
    for seed in (42, 43, 44):
        hashes = sorted({row["initial_model_sha256"] for row in rows if row["student_seed"] == seed})
        hash_audit[str(seed)] = {"unique_hashes": len(hashes), "hashes": hashes}
        if rows and len(hashes) != 1:
            errors.append({"student_seed": seed, "error": "initial hashes differ"})
    payload = {"status": "complete" if len(rows) == 12 and not errors else "failed",
               "dataset": "CUB_imsize224", "ipc": 3, "student_seeds": [42, 43, 44],
               "lgm_generation_seed": None, "random_selection_seeds": [0, 1, 2],
               "protocol": "standard_protocol_v2_soft", "initial_model_hash_audit": hash_audit,
               "groups": groups, "rows": rows, "errors": errors}
    output = args.root / "summary/cub_lgm_random_soft_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": payload["status"], "rows": len(rows), "errors": errors}, indent=2))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
