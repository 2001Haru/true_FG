"""Summarize ImageNet-v2, random-v2-LR and standard-v1 A/B common-label arms."""

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
    parser.add_argument("--a-root", required=True, type=Path)
    parser.add_argument("--cross-view-root", required=True, type=Path)
    args = parser.parse_args()
    rows = []
    errors = []
    for seed in (42, 43, 44):
        paths = {
            "imagenet_v2_a": args.a_root / f"results/entropy_t20/ipc5_sseed{seed}.json",
            "imagenet_v2_b": args.cross_view_root / f"results/ipc5_sseed{seed}.json",
            "random_v2_lrs_a": args.root / f"results/random_v2_lrs/a_image_a_label/ipc5_sseed{seed}.json",
            "random_v2_lrs_b": args.root / f"results/random_v2_lrs/b_image_a_label/ipc5_sseed{seed}.json",
            "standard_v1_a": args.root / f"results/standard_v1/a_image_a_label/ipc5_sseed{seed}.json",
            "standard_v1_b": args.root / f"results/standard_v1/b_image_a_label/ipc5_sseed{seed}.json",
        }
        try:
            payloads = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
            for payload in payloads.values():
                audit_payload(payload, 100, 3333)
            hashes = {
                name: payloads[name]["initial_model_sha256"]
                for name in ("random_v2_lrs_a", "random_v2_lrs_b", "standard_v1_a", "standard_v1_b")
            }
            if len(set(hashes.values())) != 1:
                raise RuntimeError(f"random initial-state hash mismatch: {hashes}")
            row = {"student_seed": seed, "random_initial_model_sha256": next(iter(hashes.values()))}
            for name, payload in payloads.items():
                row[f"{name}_best"] = payload["best_top1"]
                row[f"{name}_final"] = payload["final_epoch_top1"]
                row[f"{name}_result"] = str(paths[name].resolve())
            for protocol in ("imagenet_v2", "random_v2_lrs", "standard_v1"):
                row[f"{protocol}_a_minus_b_best"] = row[f"{protocol}_a_best"] - row[f"{protocol}_b_best"]
                row[f"{protocol}_a_minus_b_final"] = row[f"{protocol}_a_final"] - row[f"{protocol}_b_final"]
            for view in ("a", "b"):
                row[f"random_init_effect_{view}_best"] = row[f"random_v2_lrs_{view}_best"] - row[f"imagenet_v2_{view}_best"]
                row[f"random_init_effect_{view}_final"] = row[f"random_v2_lrs_{view}_final"] - row[f"imagenet_v2_{view}_final"]
                row[f"v1_optimizer_effect_{view}_best"] = row[f"standard_v1_{view}_best"] - row[f"random_v2_lrs_{view}_best"]
                row[f"v1_optimizer_effect_{view}_final"] = row[f"standard_v1_{view}_final"] - row[f"random_v2_lrs_{view}_final"]
            rows.append(row)
        except Exception as error:
            errors.append({"student_seed": seed, "error": str(error)})
    summary = {"count": len(rows)}
    if rows:
        fields = [key for key in rows[0] if key.endswith("_best") or key.endswith("_final")]
        summary.update({field: stats(row[field] for row in rows) for field in fields})
    payload = {
        "status": "complete" if len(rows) == 3 and not errors else "failed",
        "dataset": "A_imsize224", "ipc": 5, "teacher_seed": 42,
        "student_seeds": [42, 43, 44], "common_labels": "A original H20 IPC5 FKD",
        "protocols": {
            "imagenet_v2": "ImageNet-V1 init, backbone/head 1e-4/1e-3, full cosine",
            "random_v2_lrs": "random init only; backbone/head 1e-4/1e-3, full cosine",
            "standard_v1": "random init, uniform 1e-3, eta2 half-cosine",
        },
        "random_initial_hashes_equal_across_views_and_protocols": not errors and len(rows) == 3,
        "summary": summary, "rows": rows, "errors": errors,
    }
    output = args.root / "summary/student_init_protocol_ablation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": payload["status"], "rows": len(rows), "errors": len(errors), "output": str(output.resolve())}))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
