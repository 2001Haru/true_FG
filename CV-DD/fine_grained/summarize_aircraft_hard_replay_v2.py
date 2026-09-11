"""Summarize Aircraft standard-v2 hard-label replay matrix and paired soft deltas."""

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
    for path in sorted((args.root / "results").glob("**/*.json")):
        if path.name.endswith(".lock"):
            continue
        try:
            hard = json.loads(path.read_text(encoding="utf-8"))
            meta = hard["hard_replay_v2"]
            soft = json.loads(Path(meta["source_soft_result"]).read_text(encoding="utf-8"))
            row = {"method": meta["method"], "source_seed": meta["source_seed"],
                   "ipc": meta["ipc"], "student_seed": meta["student_seed"],
                   "hard_best": hard["best_top1"], "hard_final": hard["final_epoch_top1"],
                   "soft_best": soft["best_top1"], "soft_final": soft["final_epoch_top1"],
                   "hard_minus_soft_best": hard["best_top1"] - soft["best_top1"],
                   "hard_minus_soft_final": hard["final_epoch_top1"] - soft["final_epoch_top1"],
                   "best_epoch": hard["best_epoch"], "initial_model_sha256": hard["initial_model_sha256"],
                   "result": str(path.resolve())}
            rows.append(row)
        except Exception as error:
            errors.append({"path": str(path), "error": str(error)})
    expected = 78
    groups = []
    keys = sorted({(row["method"], row["ipc"]) for row in rows})
    for method, ipc in keys:
        selected = [row for row in rows if row["method"] == method and row["ipc"] == ipc]
        groups.append({"method": method, "ipc": ipc, "count": len(selected),
                       "hard_best": stats(row["hard_best"] for row in selected),
                       "hard_final": stats(row["hard_final"] for row in selected),
                       "soft_best": stats(row["soft_best"] for row in selected),
                       "soft_final": stats(row["soft_final"] for row in selected),
                       "hard_minus_soft_best": stats(row["hard_minus_soft_best"] for row in selected),
                       "hard_minus_soft_final": stats(row["hard_minus_soft_final"] for row in selected),
                       "best_epochs": [row["best_epoch"] for row in selected]})
    hash_audit = {}
    for seed in (42, 43, 44):
        hashes = sorted({row["initial_model_sha256"] for row in rows if row["student_seed"] == seed})
        hash_audit[str(seed)] = {"unique_hashes": len(hashes), "hashes": hashes}
        if rows and len(hashes) != 1:
            errors.append({"student_seed": seed, "error": f"initial model hashes differ: {hashes}"})
    payload = {"status": "complete" if len(rows) == expected and not errors else "failed",
               "dataset": "A_imsize224", "expected_unique_results": expected,
               "logical_results_including_reused_h20_A": 81,
               "protocol": "standard_v2 with only FKD soft logits replaced by hard CutMix CE",
               "initial_model_hash_audit": hash_audit,
               "groups": groups, "rows": rows, "errors": errors}
    output = args.root / "summary/aircraft_hard_replay_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": payload["status"], "rows": len(rows), "errors": errors}, indent=2))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
