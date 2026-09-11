"""Summarize CoDA VAE gseed0 Aircraft standard Soft-v2 results."""

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
    rows, groups, errors = [], [], []
    for ipc in (1, 3, 5):
        selected = []
        for seed in (42, 43, 44):
            path = args.root / f"results/ipc{ipc}_sseed{seed}.json"
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
                if result.get("coda_soft_v2", {}).get("status") != "complete":
                    raise RuntimeError("missing CoDA Soft-v2 audit")
                row = {"ipc": ipc, "student_seed": seed, "best_top1": result["best_top1"],
                       "final_top1": result["final_epoch_top1"], "best_epoch": result["best_epoch"],
                       "initial_model_sha256": result["initial_model_sha256"], "result": str(path.resolve())}
                rows.append(row); selected.append(row)
            except Exception as error:
                errors.append({"ipc": ipc, "student_seed": seed, "error": str(error)})
        if selected:
            groups.append({"ipc": ipc, "count": len(selected),
                           "best_top1": stats(row["best_top1"] for row in selected),
                           "final_top1": stats(row["final_top1"] for row in selected),
                           "best_epochs": [row["best_epoch"] for row in selected]})
    payload = {"status": "complete" if len(rows) == 9 and not errors else "failed",
               "dataset": "A_imsize224", "method": "CoDA", "feature_space": "vae",
               "generation_seed": 0, "student_seeds": [42, 43, 44],
               "protocol": "standard_protocol_v2_soft", "groups": groups, "rows": rows, "errors": errors}
    output = args.root / "summary/coda_soft_v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": payload["status"], "rows": len(rows), "errors": errors}, indent=2))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
