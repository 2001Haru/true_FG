"""Summarize Aircraft RandomReal IPC5/10 Soft-v2 with 6000 consumed FKD batches."""

import argparse
import json
import statistics
from pathlib import Path


RSEEDS = (0, 1, 2)
SSEEDS = (42, 43, 44)
CONFIG = {5: (240, 25), 10: (120, 50)}


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
    for ipc, (epochs, batches_per_epoch) in CONFIG.items():
        for rseed in RSEEDS:
            for sseed in SSEEDS:
                path = a.root / f"results/ipc{ipc}/rseed{rseed}/sseed{sseed}.json"
                ref_path = a.scaling_root / f"results/rseed{rseed}/ipc{ipc}_sseed{sseed}.json"
                row, reference = json.loads(path.read_text()), json.loads(ref_path.read_text())
                if row.get("epochs") != epochs or row.get("batch_size") != 20 or row.get("gradient_accumulation_steps") != 2:
                    raise RuntimeError(f"budget mismatch: {path}")
                if row.get("optimizer_updates_per_epoch") != batches_per_epoch or row.get("total_optimizer_updates") != 6000:
                    raise RuntimeError(f"consumed-batch mismatch: {path}")
                if row.get("scheduler") != "cosine_annealing" or row.get("scheduler_t_max") != epochs:
                    raise RuntimeError(f"scheduler mismatch: {path}")
                if any(abs(float(value)) > 1e-12 for value in row.get("final_scheduler_lrs", [])):
                    raise RuntimeError(f"final LR is not zero: {path}")
                for key in ("synthetic_data_path", "fkd_path", "student_seed", "initial_model_sha256"):
                    if row.get(key) != reference.get(key):
                        raise RuntimeError(f"reference pairing mismatch {key}: {path}")
                if row.get("student_initialization") != "imagenet-v1" or row.get("temperature") != 20:
                    raise RuntimeError(f"Student protocol mismatch: {path}")
                rows[(ipc, rseed, sseed)], full[(ipc, rseed, sseed)] = row, reference
    ipc3 = {sseed: json.loads((a.ipc3_root / f"ipc3_sseed{sseed}.json").read_text()) for sseed in SSEEDS}
    for ipc in CONFIG:
        for sseed in SSEEDS:
            if rows[(ipc, 0, sseed)]["initial_model_sha256"] != ipc3[sseed]["initial_model_sha256"]:
                raise RuntimeError(f"IPC{ipc}/IPC3 initialization mismatch: sseed{sseed}")

    result = {
        "status": "complete", "protocol": "random_real_ipc5_10_soft_v2_budget6000_v1",
        "ipcs": [5, 10], "selection_seeds": list(RSEEDS), "student_seeds": list(SSEEDS),
        "physical_batches_per_run": 6000, "views_per_run": 120000,
        "optimizer_steps_with_accumulation2": 3000, "new_fkd_sets": 0,
        "teacher_queries": 0, "new_student_runs": 18, "by_ipc": {},
    }
    for ipc, (epochs, _) in CONFIG.items():
        keys = [(ipc, rseed, sseed) for rseed in RSEEDS for sseed in SSEEDS]
        result["by_ipc"][str(ipc)] = {
            "epochs": epochs, "scheduler_t_max": epochs,
            "budget6000": {
                "best_top1": stats(rows[key]["best_top1"] for key in keys),
                "final_top1": stats(rows[key]["final_epoch_top1"] for key in keys),
                "by_selection_seed": {
                    str(rseed): {
                        "best_top1": stats(rows[(ipc, rseed, sseed)]["best_top1"] for sseed in SSEEDS),
                        "final_top1": stats(rows[(ipc, rseed, sseed)]["final_epoch_top1"] for sseed in SSEEDS),
                    } for rseed in RSEEDS
                },
            },
            "paired_budget6000_minus_full": {
                "best_top1": stats(rows[key]["best_top1"] - full[key]["best_top1"] for key in keys),
                "final_top1": stats(rows[key]["final_epoch_top1"] - full[key]["final_epoch_top1"] for key in keys),
            },
            "full_reference": {
                "best_top1": stats(full[key]["best_top1"] for key in keys),
                "final_top1": stats(full[key]["final_epoch_top1"] for key in keys),
                "physical_batches": 400 * CONFIG[ipc][1], "views": 400 * CONFIG[ipc][1] * 20,
            },
        }
    result["paired_same_budget_ipc10_minus_ipc5"] = {
        "best_top1": stats(rows[(10, r, s)]["best_top1"] - rows[(5, r, s)]["best_top1"] for r in RSEEDS for s in SSEEDS),
        "final_top1": stats(rows[(10, r, s)]["final_epoch_top1"] - rows[(5, r, s)]["final_epoch_top1"] for r in RSEEDS for s in SSEEDS),
    }
    for ipc in CONFIG:
        result[f"paired_same_budget_ipc{ipc}_rseed0_minus_ipc3_rseed0"] = {
            "best_top1": stats(rows[(ipc, 0, s)]["best_top1"] - ipc3[s]["best_top1"] for s in SSEEDS),
            "final_top1": stats(rows[(ipc, 0, s)]["final_epoch_top1"] - ipc3[s]["final_epoch_top1"] for s in SSEEDS),
        }
    result["ipc3_rseed0_reference"] = {
        "best_top1": stats(ipc3[s]["best_top1"] for s in SSEEDS),
        "final_top1": stats(ipc3[s]["final_epoch_top1"] for s in SSEEDS),
        "physical_batches": 6000, "views": 120000, "epochs": 400,
    }
    output = a.root / "summary/summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
