"""Summarize the closed-form axis/k transfer matrix."""

import argparse
import json
import statistics
from pathlib import Path


SEEDS = (42, 43, 44)
GROUPS = {
    "SC_ipc3_k3": ("f1",),
    "CUB_ipc3_k2": ("u6", "f1"),
    "SC_ipc3_k2": ("u6", "f1"),
    "A_ipc1_k3": ("r0", "uk", "f1"),
    "CUB_ipc1_k2": ("r0", "uk", "f1"),
    "SC_ipc1_k2": ("r0", "uk", "f1"),
}


def stats(values):
    values = list(map(float, values))
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
            "values": values, "positive_count": sum(value > 0 for value in values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--old-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    groups, comparisons, raw = {}, {}, {}
    for name, arms in GROUPS.items():
        raw[name] = {}
        for arm in arms:
            values = {seed: json.loads((args.root / "results" / f"{name}_{arm}_sseed{seed}.json").read_text())
                      for seed in SEEDS}
            raw[name][arm] = values
            groups[f"{name}_{arm}"] = {
                "best": stats(values[seed]["best_top1"] for seed in SEEDS),
                "final": stats(values[seed]["final_epoch_top1"] for seed in SEEDS),
                "best_epochs": [values[seed]["best_epoch"] for seed in SEEDS],
            }
        rows = raw[name]
        if {"r0", "uk", "f1"} <= rows.keys():
            comparisons[name] = {
                key: stats(rows[left][seed]["final_epoch_top1"] - rows[right][seed]["final_epoch_top1"]
                           for seed in SEEDS)
                for key, left, right in (
                    ("instance_uk_minus_r0", "uk", "r0"),
                    ("encoding_uk_minus_f1", "uk", "f1"),
                    ("net_f1_minus_r0", "f1", "r0"),
                )
            }
        elif {"u6", "f1"} <= rows.keys():
            comparisons[name] = {
                "encoding_u6_minus_f1": stats(
                    rows["u6"][seed]["final_epoch_top1"] - rows["f1"][seed]["final_epoch_top1"]
                    for seed in SEEDS)}
    prior = json.loads(args.old_summary.read_text())
    old_column = prior["groups"]["SC_imsize224_f1"]["final"]["values"]
    comparisons.setdefault("SC_ipc3_k3", {})["row_minus_old_column_f1"] = stats(
        raw["SC_ipc3_k3"]["f1"][seed]["final_epoch_top1"] - old_column[index]
        for index, seed in enumerate(SEEDS))
    output = {
        "status": "complete", "protocol": "fg_closed_form_transfer_v1",
        "phi": .106, "tau": .6,
        "teacher_final_top1": {"A_imsize224": 83.6784, "CUB_imsize224": 71.25,
                               "SC_imsize224": 85.62},
        "groups": groups, "comparisons": comparisons,
        "expected_fkd": 14, "observed_fkd": len(list((args.root / "stage" / "fkd").glob("*/*/relabel_manifest.json"))),
        "expected_students": 42, "observed_students": len(list((args.root / "results").glob("*.json"))),
        "preflight": json.loads((args.root / "audits" / "preflight.json").read_text()),
    }
    if output["observed_fkd"] != output["expected_fkd"] or output["observed_students"] != output["expected_students"]:
        raise RuntimeError((output["observed_fkd"], output["observed_students"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
