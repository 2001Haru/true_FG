"""Summarize Aircraft IPC3 retarget-compression Soft-v2 diagnostics."""

import argparse
import json
import statistics
from pathlib import Path


METHODS = ("isotropic158", "vertical112", "horizontal112", "bbox_retarget112", "attention_retarget112")
SEEDS = (42, 43, 44)


def stats(values):
    values = list(map(float, values))
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "values": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--r0-results", required=True, type=Path)
    args = parser.parse_args()
    construction = json.loads((args.root / "construction/construction_manifest.json").read_text())
    rows, errors = {}, []
    for method in METHODS:
        for seed in SEEDS:
            path = args.root / f"results/{method}/ipc3_sseed{seed}.json"
            try:
                row = json.loads(path.read_text())
                assert row["training_target"] == "fkd_soft_label" and row["mix_type"] == "cutmix"
                assert row["student_seed"] == seed and row["epochs"] == 400
                rows[(method, seed)] = row
            except Exception as error:
                errors.append({"path": str(path), "error": repr(error)})
    r0 = {}
    for seed in SEEDS:
        path = args.r0_results / f"ipc3_sseed{seed}.json"
        try:
            r0[seed] = json.loads(path.read_text())
        except Exception as error:
            errors.append({"path": str(path), "error": repr(error)})
    groups = {}
    for method in METHODS:
        groups[method] = {
            "best_top1": stats(rows[(method, seed)]["best_top1"] for seed in SEEDS),
            "final_top1": stats(rows[(method, seed)]["final_epoch_top1"] for seed in SEEDS),
            "paired_minus_r0_best": stats(rows[(method, seed)]["best_top1"] - r0[seed]["best_top1"] for seed in SEEDS),
            "paired_minus_r0_final": stats(rows[(method, seed)]["final_epoch_top1"] - r0[seed]["final_epoch_top1"] for seed in SEEDS),
            "best_epochs": [rows[(method, seed)]["best_epoch"] for seed in SEEDS],
        }
    for seed in SEEDS:
        hashes = {r0[seed]["initial_model_sha256"]} | {rows[(method, seed)]["initial_model_sha256"] for method in METHODS}
        if len(hashes) != 1:
            errors.append({"student_seed": seed, "error": "initial model hash mismatch"})
    result = {"status": "complete" if len(rows) == 15 and len(r0) == 3 and not errors else "failed",
              "protocol": "aircraft_ipc3_retarget_compression_soft_v2_v1", "dataset": "A_imsize224", "ipc": 3,
              "condition": "fullframe+flip+CutMix; replayed identical metadata; actual-view Teacher42 T20 KL",
              "groups": groups,
              "r0": {"best_top1": stats(r0[s]["best_top1"] for s in SEEDS),
                     "final_top1": stats(r0[s]["final_epoch_top1"] for s in SEEDS)},
              "construction": {key: construction[key] for key in
                               ("encoded_storage", "decoded_storage", "retarget_encoding", "retarget_decoding",
                                "bbox_audit", "attention_density_audit", "outputs")},
              "new_fkd_sets": 5, "new_student_runs": 15, "errors": errors}
    output = args.root / "summary/summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "groups": groups, "errors": errors}, indent=2))
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
