"""Summarize RandomReal against A and D under both ImageNette protocols."""

import argparse
import json
import os
import statistics
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def value(payload, suffix):
    if suffix == "best":
        return payload["best_top1"]
    return payload["final_top1"] if "final_top1" in payload else payload["final_epoch_top1"]


def stats(values):
    values = list(map(float, values))
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "values": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--ab-root", required=True, type=Path)
    args = parser.parse_args()
    rows, errors = [], []
    for ipc in (10, 50):
        for seed in (42, 43, 44):
            paths = {
                "cvdd_a": args.ab_root / f"results/a_image/ipc{ipc}_sseed{seed}.json",
                "cvdd_d": args.root / f"results/current_cvdd/d_rded/ipc{ipc}_sseed{seed}.json",
                "cvdd_random": args.root / f"results/current_cvdd/random_real/ipc{ipc}_sseed{seed}.json",
                "native_a": args.root / f"results/rded_native/a_original/ipc{ipc}_sseed{seed}/result.json",
                "native_d": args.root / f"results/rded_native/d_rded/ipc{ipc}_sseed{seed}/result.json",
                "native_random": args.root / f"results/rded_native/random_real/ipc{ipc}_sseed{seed}/result.json",
            }
            try:
                payloads = {key: load(path) for key, path in paths.items()}
                for protocol in ("cvdd", "native"):
                    hashes = {payloads[f"{protocol}_{arm}"]["initial_model_sha256"] for arm in ("a", "d", "random")}
                    if len(hashes) != 1:
                        raise RuntimeError(f"{protocol} A/D/Random initial hashes differ")
                row = {"ipc": ipc, "student_seed": seed}
                for suffix in ("best", "final"):
                    for key, payload in payloads.items():
                        row[f"{key}_{suffix}"] = value(payload, suffix)
                    for protocol in ("cvdd", "native"):
                        row[f"{protocol}_a_minus_random_{suffix}"] = row[f"{protocol}_a_{suffix}"] - row[f"{protocol}_random_{suffix}"]
                        row[f"{protocol}_d_minus_random_{suffix}"] = row[f"{protocol}_d_{suffix}"] - row[f"{protocol}_random_{suffix}"]
                rows.append(row)
            except Exception as error:
                errors.append({"ipc": ipc, "student_seed": seed, "error": str(error)})
    groups = []
    fields = [f"{p}_{a}" for p in ("cvdd", "native") for a in ("a", "d", "random", "a_minus_random", "d_minus_random")]
    for ipc in (10, 50):
        selected = [row for row in rows if row["ipc"] == ipc]
        group = {"ipc": ipc, "count": len(selected)}
        for field in fields:
            for suffix in ("best", "final"):
                group[f"{field}_{suffix}"] = stats(row[f"{field}_{suffix}"] for row in selected)
        groups.append(group)
    payload = {"status": "complete" if len(rows) == 6 and not errors else "failed",
               "dataset": "imagenet-nette", "selection_seed": 42, "validation_images": 3925,
               "groups": groups, "rows": rows, "errors": errors}
    output = args.root / "summary/imagenette_random_protocol.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": payload["status"], "rows": len(rows), "errors": errors, "output": str(output.resolve())}, indent=2))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
