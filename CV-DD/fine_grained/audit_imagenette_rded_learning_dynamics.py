"""Audit existing A/D learning curves around native-RDED update budgets."""

import argparse
import json
import os
import re
import statistics
from pathlib import Path


PATTERN = re.compile(r"TEST Iter (\d+):.*Top-1 err = ([0-9.]+)")
SEEDS = (42, 43, 44)


def curve(path):
    result = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = PATTERN.search(line)
        if match:
            result[int(match.group(1))] = 100.0 - float(match.group(2))
    return result


def stats(values):
    values = list(map(float, values))
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "values": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--ab-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = {"status": "complete", "dataset": "imagenet-nette", "student_seeds": list(SEEDS), "ipc": {}}
    for ipc, native_updates in ((10, 600), (50, 1500)):
        curves = {"a": {}, "d": {}}
        for seed in SEEDS:
            curves["a"][seed] = curve(args.ab_root / f"logs/eval_a_image_ipc{ipc}_sseed{seed}.log")
            curves["d"][seed] = curve(args.root / f"logs/current_d_ipc{ipc}_sseed{seed}.log")
        common = sorted(set.intersection(*(set(one) for arm in curves.values() for one in arm.values())))
        rows = []
        for epoch_index in common:
            a = [curves["a"][seed][epoch_index] for seed in SEEDS]
            d = [curves["d"][seed][epoch_index] for seed in SEEDS]
            delta = [dv - av for av, dv in zip(a, d)]
            rows.append({
                "log_epoch_index": epoch_index, "completed_epochs": epoch_index + 1,
                "optimizer_updates": (epoch_index + 1) * ipc,
                "a_top1": stats(a), "d_top1": stats(d), "d_minus_a": stats(delta),
            })
        before = max((row for row in rows if row["optimizer_updates"] <= native_updates), key=lambda row: row["optimizer_updates"])
        after = min((row for row in rows if row["optimizer_updates"] >= native_updates), key=lambda row: row["optimizer_updates"])
        native_rows = []
        for epoch in (250, 260, 270, 280, 290, 300):
            a_values, d_values = [], []
            for seed in SEEDS:
                histories = {}
                for arm in ("a_original", "d_rded"):
                    path = args.root / f"results/rded_native/{arm}/ipc{ipc}_sseed{seed}/result.json"
                    history = json.loads(path.read_text(encoding="utf-8"))["history"]
                    histories[arm] = {item["epoch"]: item["evaluation"]["top1"] for item in history if item["evaluation"]}
                a_values.append(histories["a_original"][epoch])
                d_values.append(histories["d_rded"][epoch])
            native_rows.append({"epoch": epoch, "optimizer_updates": epoch * (2 if ipc == 10 else 5),
                                "a_top1": stats(a_values), "d_top1": stats(d_values),
                                "d_minus_a": stats(dv-av for av, dv in zip(a_values, d_values))})
        result["ipc"][str(ipc)] = {
            "native_total_optimizer_updates": native_updates,
            "exact_cvdd_aligned_checkpoint_available": before["optimizer_updates"] == native_updates,
            "cvdd_bracketing_checkpoints": {"before": before, "after": after},
            "cvdd_curve": rows, "native_late_curve": native_rows,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"status": "complete", "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
