"""Summarize the Aircraft DeCO-style FG-region Soft RRC scale sweep."""
import argparse
import json
import statistics
from pathlib import Path


SEEDS = (42, 43, 44)
ARMS = ("rrc_008_100", "rrc_010_070", "rrc_030_100", "rrc_050_100")
RANGES = {
    "rrc_008_100": (0.08, 1.0),
    "rrc_010_070": (0.1, 0.7),
    "rrc_030_100": (0.3, 1.0),
    "rrc_050_100": (0.5, 1.0),
}
METRICS = ("best_top1", "final_epoch_top1")


def stats(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "values": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--baseline-results", required=True, type=Path)
    args = parser.parse_args()
    rows = {
        arm: [json.loads(((args.baseline_results if arm == "rrc_008_100" else args.root / f"results/{arm}") /
                          f"ipc3_sseed{seed}.json").read_text()) for seed in SEEDS]
        for arm in ARMS
    }
    for arm in ARMS[1:]:
        manifest = json.loads((args.root / f"fkd/{arm}/ipc3_bs20_ipc3/relabel_manifest.json").read_text())
        assert manifest["status"] == "complete" and manifest["mix_type"] == "cutmix"
        assert abs(manifest["min_scale_crops"] - RANGES[arm][0]) < 1e-12
        assert abs(manifest["max_scale_crops"] - RANGES[arm][1]) < 1e-12
        for candidate, reference, seed in zip(rows[arm], rows["rrc_008_100"], SEEDS):
            assert candidate["student_seed"] == reference["student_seed"] == seed
            assert candidate["initial_model_sha256"] == reference["initial_model_sha256"]
            assert candidate["epochs"] == reference["epochs"] == 400
            assert candidate["temperature"] == reference["temperature"] == 20
            assert candidate["mix_type"] == reference["mix_type"] == "cutmix"
    groups = {arm: {metric: stats([row[metric] for row in rows[arm]]) for metric in METRICS} for arm in ARMS}
    contrasts = {
        f"{arm} minus rrc_008_100": {
            metric: stats([a[metric] - b[metric] for a, b in zip(rows[arm], rows["rrc_008_100"])])
            for metric in METRICS}
        for arm in ARMS[1:]
    }
    result = {"status": "complete", "protocol": "deco_fg_regions_soft_v2_rrc_scale_sweep_v1",
              "rrc_ranges": {arm: list(RANGES[arm]) for arm in ARMS}, "groups": groups,
              "paired_contrasts": contrasts, "student_seeds": list(SEEDS),
              "new_fkd_sets": 3, "new_student_runs": 9}
    output = args.root / "summary/summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
