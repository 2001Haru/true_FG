"""Summarize the full-RDED × downstream-protocol ImageNette matrix."""

import argparse
import json
import os
import statistics
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values):
    values = list(map(float, values))
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "values": values}


def metric(payload, name):
    if name == "best":
        return payload["best_top1"]
    return payload.get("final_top1", payload.get("final_epoch_top1"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--ab-root", required=True, type=Path)
    parser.add_argument("--c-root", required=True, type=Path)
    args = parser.parse_args()
    rows, errors = [], []
    for ipc in (10, 50):
        for seed in (42, 43, 44):
            paths = {
                "cvdd_a": args.ab_root / f"results/a_image/ipc{ipc}_sseed{seed}.json",
                "cvdd_d": args.root / f"results/current_cvdd/d_rded/ipc{ipc}_sseed{seed}.json",
                "native_a": args.root / f"results/rded_native/a_original/ipc{ipc}_sseed{seed}/result.json",
                "native_d": args.root / f"results/rded_native/d_rded/ipc{ipc}_sseed{seed}/result.json",
                "cvdd_c": args.c_root / f"results/ipc{ipc}_sseed{seed}.json",
            }
            try:
                payloads = {key: load(path) for key, path in paths.items()}
                if payloads["cvdd_a"]["initial_model_sha256"] != payloads["cvdd_d"]["initial_model_sha256"]:
                    raise RuntimeError("CV-DD A/D initial hashes differ")
                if payloads["native_a"]["initial_model_sha256"] != payloads["native_d"]["initial_model_sha256"]:
                    raise RuntimeError("native A/D initial hashes differ")
                row = {"ipc": ipc, "student_seed": seed, "paths": {key: str(value.resolve()) for key, value in paths.items()}}
                for kind, payload in payloads.items():
                    row[f"{kind}_best"] = metric(payload, "best")
                    row[f"{kind}_final"] = metric(payload, "final")
                for suffix in ("best", "final"):
                    row[f"cvdd_d_minus_a_{suffix}"] = row[f"cvdd_d_{suffix}"] - row[f"cvdd_a_{suffix}"]
                    row[f"native_d_minus_a_{suffix}"] = row[f"native_d_{suffix}"] - row[f"native_a_{suffix}"]
                    row[f"interaction_{suffix}"] = row[f"native_d_minus_a_{suffix}"] - row[f"cvdd_d_minus_a_{suffix}"]
                    row[f"cvdd_d_minus_c_{suffix}"] = row[f"cvdd_d_{suffix}"] - row[f"cvdd_c_{suffix}"]
                rows.append(row)
            except Exception as error:
                errors.append({"ipc": ipc, "student_seed": seed, "error": str(error)})
    groups = []
    fields = [
        "cvdd_a", "cvdd_d", "native_a", "native_d", "cvdd_c",
        "cvdd_d_minus_a", "native_d_minus_a", "interaction", "cvdd_d_minus_c",
    ]
    for ipc in (10, 50):
        selected = [row for row in rows if row["ipc"] == ipc]
        group = {"ipc": ipc, "count": len(selected)}
        for field in fields:
            for suffix in ("best", "final"):
                group[f"{field}_{suffix}"] = stats(row[f"{field}_{suffix}"] for row in selected) if len(selected) >= 2 else None
        groups.append(group)
    payload = {
        "status": "complete" if len(rows) == 6 and not errors else "failed",
        "dataset": "imagenet-nette", "validation_images": 3925,
        "primary_comparison": "D minus A within protocol; interaction=(D-A)_native-(D-A)_cvdd",
        "context_comparison": "current-CVDD full RDED D minus same-source mosaic C",
        "groups": groups, "rows": rows, "errors": errors,
    }
    output = args.root / "summary/imagenette_full_rded_protocol.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": payload["status"], "rows": len(rows), "errors": errors, "output": str(output.resolve())}, indent=2))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
