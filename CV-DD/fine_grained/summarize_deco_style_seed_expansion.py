"""Summarize three construction seeds for RRC-on Soft DeCO-style regions."""
import argparse
import json
import statistics
from pathlib import Path


SEEDS = (42, 43, 44)
METHODS = ("random_regions", "fg_regions")
METRICS = ("best_top1", "final_epoch_top1")


def stats(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "values": values}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--seed0-root", required=True, type=Path)
    p.add_argument("--r0-results-root", required=True, type=Path)
    a = p.parse_args()
    rows = {method: {} for method in METHODS}
    r0 = {}
    for cseed in (0, 1, 2):
        for sseed in SEEDS:
            for method in METHODS:
                path = (a.seed0_root / f"results/{method}/ipc3_sseed{sseed}.json" if cseed == 0
                        else a.root / f"results/cseed{cseed}/{method}/ipc3_sseed{sseed}.json")
                rows[method][cseed, sseed] = json.loads(path.read_text())
            r0[cseed, sseed] = json.loads((a.r0_results_root / f"rseed{cseed}/ipc3_sseed{sseed}.json").read_text())
    for method in METHODS:
        for key, candidate in rows[method].items():
            reference = r0[key]
            assert candidate["student_seed"] == reference["student_seed"] == key[1]
            assert candidate["temperature"] == 20 and candidate["epochs"] == 400
            assert candidate["mix_type"] == "cutmix" and not candidate["fkd_hard_label"]
    groups = {
        "r0": {m: stats([r0[k][m] for k in sorted(r0)]) for m in METRICS},
        **{method: {m: stats([rows[method][k][m] for k in sorted(rows[method])]) for m in METRICS}
           for method in METHODS},
    }
    by_construction_seed = {}
    for cseed in (0, 1, 2):
        by_construction_seed[str(cseed)] = {
            name: {m: stats([source[cseed, s][m] for s in SEEDS]) for m in METRICS}
            for name, source in (("r0", r0), ("random_regions", rows["random_regions"]), ("fg_regions", rows["fg_regions"]))}
    contrasts = {}
    for left, right, left_rows, right_rows in (
        ("random_regions", "r0", rows["random_regions"], r0),
        ("fg_regions", "r0", rows["fg_regions"], r0),
        ("fg_regions", "random_regions", rows["fg_regions"], rows["random_regions"]),
    ):
        contrasts[f"{left} minus {right}"] = {
            m: stats([left_rows[k][m] - right_rows[k][m] for k in sorted(left_rows)]) for m in METRICS}
    result = {"status": "complete", "protocol": "controlled_deco_style_regions_rrc_soft_three_construction_seeds_v1",
              "construction_seed_mapping": {"0": 20260914, "1": 20260915, "2": 20260916},
              "r0_selection_seed_mapping": {"0": 0, "1": 1, "2": 2},
              "student_seeds": list(SEEDS), "groups": groups, "by_construction_seed": by_construction_seed,
              "paired_contrasts": contrasts, "new_fkd_sets": 4, "new_student_runs": 12,
              "reused_r0_student_runs": 9}
    out = a.root / "summary/summary.json";out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n");print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
