"""Summarize ImageNette A/A, B/A and C/C native-CVDD results."""

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
    parser.add_argument("--ab-root", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    errors = []
    for ipc in (10, 50):
        for seed in (42, 43, 44):
            paths = {
                "a": args.ab_root / f"results/a_image/ipc{ipc}_sseed{seed}.json",
                "b": args.ab_root / f"results/b_image/ipc{ipc}_sseed{seed}.json",
                "c": args.root / f"results/ipc{ipc}_sseed{seed}.json",
            }
            try:
                x = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
                for payload in x.values():
                    audit_payload(payload, 10, 3925)
                if len({payload["initial_model_sha256"] for payload in x.values()}) != 1:
                    raise RuntimeError("A/B/C initial model mismatch")
                row = {"ipc": ipc, "student_seed": seed}
                for name, payload in x.items():
                    row[f"{name}_best"] = payload["best_top1"]
                    row[f"{name}_best_epoch"] = payload["best_epoch"]
                    row[f"{name}_final"] = payload["final_epoch_top1"]
                    row[f"{name}_result"] = str(paths[name].resolve())
                row["c_minus_a_best"] = row["c_best"] - row["a_best"]
                row["c_minus_a_final"] = row["c_final"] - row["a_final"]
                row["c_minus_b_best"] = row["c_best"] - row["b_best"]
                row["c_minus_b_final"] = row["c_final"] - row["b_final"]
                rows.append(row)
            except Exception as error:
                errors.append({"ipc": ipc, "student_seed": seed, "error": str(error)})
    groups = []
    for ipc in (10, 50):
        selected = [row for row in rows if row["ipc"] == ipc]
        group = {"ipc": ipc, "count": len(selected)}
        if selected:
            for field in ("a_best", "b_best", "c_best", "c_minus_a_best", "c_minus_b_best",
                          "a_final", "b_final", "c_final", "c_minus_a_final", "c_minus_b_final"):
                group[field] = stats(row[field] for row in selected)
            group["c_best_epochs"] = [row["c_best_epoch"] for row in selected]
        groups.append(group)
    payload = {
        "status": "complete" if len(rows) == 6 and not errors else "failed",
        "dataset": "imagenet-nette", "ipcs": [10, 50], "student_seeds": [42, 43, 44],
        "conditions": {"a": "A image/A label", "b": "B image/A label", "c": "C image/C label"},
        "groups": groups, "rows": rows, "errors": errors,
    }
    output = args.root / "summary/imagenette_same_source_mosaic.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": payload["status"], "rows": len(rows), "errors": len(errors), "output": str(output.resolve())}))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
