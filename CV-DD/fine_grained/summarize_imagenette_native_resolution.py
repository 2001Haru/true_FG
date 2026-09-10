"""Summarize paired ImageNette IPC10/50 A/B native-CVDD evaluations."""

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
    args = parser.parse_args()
    rows = []
    errors = []
    for ipc in (10, 50):
        for seed in (42, 43, 44):
            paths = {view: args.root / f"results/{view}/ipc{ipc}_sseed{seed}.json" for view in ("a_image", "b_image")}
            try:
                payloads = {view: json.loads(path.read_text(encoding="utf-8")) for view, path in paths.items()}
                for payload in payloads.values():
                    audit_payload(payload, 10, 3925)
                hashes = {view: payloads[view]["initial_model_sha256"] for view in payloads}
                if len(set(hashes.values())) != 1:
                    raise RuntimeError(f"A/B initial model mismatch: {hashes}")
                row = {"ipc": ipc, "student_seed": seed, "initial_model_sha256": hashes["a_image"]}
                for view, payload in payloads.items():
                    row[f"{view}_best"] = payload["best_top1"]
                    row[f"{view}_best_epoch"] = payload["best_epoch"]
                    row[f"{view}_final"] = payload["final_epoch_top1"]
                    row[f"{view}_result"] = str(paths[view].resolve())
                row["a_minus_b_best"] = row["a_image_best"] - row["b_image_best"]
                row["a_minus_b_final"] = row["a_image_final"] - row["b_image_final"]
                rows.append(row)
            except Exception as error:
                errors.append({"ipc": ipc, "student_seed": seed, "error": str(error)})
    groups = []
    for ipc in (10, 50):
        selected = [row for row in rows if row["ipc"] == ipc]
        group = {"ipc": ipc, "count": len(selected)}
        if selected:
            for field in ("a_image_best", "b_image_best", "a_minus_b_best", "a_image_final", "b_image_final", "a_minus_b_final"):
                group[field] = stats(row[field] for row in selected)
            group["a_best_epochs"] = [row["a_image_best_epoch"] for row in selected]
            group["b_best_epochs"] = [row["b_image_best_epoch"] for row in selected]
        groups.append(group)
    by_seed = {}
    for seed in (42, 43, 44):
        hashes = {row["initial_model_sha256"] for row in rows if row["student_seed"] == seed}
        by_seed[str(seed)] = {"unique_initial_hashes_across_ipc_and_views": len(hashes), "hashes": sorted(hashes)}
    payload = {
        "status": "complete" if len(rows) == 6 and not errors else "failed",
        "dataset": "imagenet-nette", "ipcs": [10, 50], "student_seeds": [42, 43, 44],
        "batch_size": 10, "gradient_accumulation_steps": 2, "micro_batch": 5,
        "optimizer": "AdamW", "learning_rate": 5e-4, "weight_decay": 0.01,
        "eta": {"10": 1, "50": 2}, "epochs": 300,
        "initial_hash_audit": by_seed, "groups": groups, "rows": rows, "errors": errors,
    }
    output = args.root / "summary/imagenette_native_resolution.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": payload["status"], "rows": len(rows), "errors": len(errors), "output": str(output.resolve())}))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
