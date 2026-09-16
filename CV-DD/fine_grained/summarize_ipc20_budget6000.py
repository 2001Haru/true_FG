"""Summarize Aircraft RandomReal IPC20 Soft-v2 with 6000 consumed FKD batches."""

import argparse
import json
import statistics
from pathlib import Path


RSEEDS = (0, 1, 2)
SSEEDS = (42, 43, 44)


def stats(values):
    values = list(map(float, values))
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "values": values}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--scaling-root", required=True, type=Path)
    p.add_argument("--ipc3-root", required=True, type=Path)
    a = p.parse_args()
    rows, full = {}, {}
    for rseed in RSEEDS:
        for sseed in SSEEDS:
            path = a.root / f"results/rseed{rseed}/ipc20_sseed{sseed}.json"
            reference_path = a.scaling_root / f"results/rseed{rseed}/ipc20_sseed{sseed}.json"
            row, reference = json.loads(path.read_text()), json.loads(reference_path.read_text())
            if row.get("epochs") != 60 or row.get("batch_size") != 20 or row.get("gradient_accumulation_steps") != 2:
                raise RuntimeError(f"budget mismatch: {path}")
            if row.get("optimizer_updates_per_epoch") != 100 or row.get("total_optimizer_updates") != 6000:
                raise RuntimeError(f"consumed-batch mismatch: {path}")
            if row.get("scheduler") != "cosine_annealing" or row.get("scheduler_t_max") != 60:
                raise RuntimeError(f"scheduler mismatch: {path}")
            if any(abs(float(value)) > 1e-12 for value in row.get("final_scheduler_lrs", [])):
                raise RuntimeError(f"final LR is not zero: {path}")
            for key in ("synthetic_data_path", "fkd_path", "student_seed", "initial_model_sha256"):
                if row.get(key) != reference.get(key):
                    raise RuntimeError(f"reference pairing mismatch {key}: {path}")
            if row.get("student_initialization") != "imagenet-v1" or row.get("temperature") != 20:
                raise RuntimeError(f"Student protocol mismatch: {path}")
            rows[(rseed, sseed)], full[(rseed, sseed)] = row, reference
    ipc3 = {sseed: json.loads((a.ipc3_root / f"ipc3_sseed{sseed}.json").read_text()) for sseed in SSEEDS}
    for sseed in SSEEDS:
        if rows[(0, sseed)]["initial_model_sha256"] != ipc3[sseed]["initial_model_sha256"]:
            raise RuntimeError(f"IPC3 initialization mismatch: sseed{sseed}")
    result = {
        "status": "complete", "protocol": "random_real_ipc20_soft_v2_budget6000_v1",
        "ipc": 20, "selection_seeds": list(RSEEDS), "student_seeds": list(SSEEDS),
        "consumed_fkd_epochs": [0, 59], "physical_batches": 6000, "views": 120000,
        "optimizer_steps_with_accumulation2": 3000, "epochs": 60, "scheduler_t_max": 60,
        "budget6000": {
            "best_top1": stats(rows[key]["best_top1"] for key in rows),
            "final_top1": stats(rows[key]["final_epoch_top1"] for key in rows),
            "by_selection_seed": {
                str(rseed): {
                    "best_top1": stats(rows[(rseed, sseed)]["best_top1"] for sseed in SSEEDS),
                    "final_top1": stats(rows[(rseed, sseed)]["final_epoch_top1"] for sseed in SSEEDS),
                } for rseed in RSEEDS
            },
        },
        "paired_budget6000_minus_full40000": {
            "best_top1": stats(rows[key]["best_top1"] - full[key]["best_top1"] for key in rows),
            "final_top1": stats(rows[key]["final_epoch_top1"] - full[key]["final_epoch_top1"] for key in rows),
        },
        "paired_ipc20_rseed0_minus_ipc3_rseed0_at_6000_batches": {
            "best_top1": stats(rows[(0, sseed)]["best_top1"] - ipc3[sseed]["best_top1"] for sseed in SSEEDS),
            "final_top1": stats(rows[(0, sseed)]["final_epoch_top1"] - ipc3[sseed]["final_epoch_top1"] for sseed in SSEEDS),
        },
        "ipc3_rseed0_reference": {
            "best_top1": stats(ipc3[sseed]["best_top1"] for sseed in SSEEDS),
            "final_top1": stats(ipc3[sseed]["final_epoch_top1"] for sseed in SSEEDS),
            "physical_batches": 6000, "views": 120000, "epochs": 400,
        },
        "ipc20_full_reference": {
            "best_top1": stats(full[key]["best_top1"] for key in full),
            "final_top1": stats(full[key]["final_epoch_top1"] for key in full),
            "physical_batches": 40000, "views": 800000, "epochs": 400,
        },
        "new_fkd_sets": 0, "teacher_queries": 0, "new_student_runs": 9,
    }
    output = a.root / "summary/summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
