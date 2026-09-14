"""Summarize Hard-label RRC factorial for controlled DeCO-style mosaics."""
import argparse
import json
import statistics
from pathlib import Path


SEEDS = (42, 43, 44)
VIEWS = ("fullframe_cutmix", "rrc_cutmix")
METHODS = ("r0", "random_regions", "fg_regions")
METRICS = ("best_top1", "final_epoch_top1")


def stats(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "values": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--r0-full-results", required=True, type=Path)
    parser.add_argument("--r0-rrc-results", required=True, type=Path)
    args = parser.parse_args()
    rows = {}
    for view in VIEWS:
        rows[view] = {}
        for method in METHODS:
            base = ({"fullframe_cutmix": args.r0_full_results, "rrc_cutmix": args.r0_rrc_results}[view]
                    if method == "r0" else args.root / f"results/{view}/{method}")
            rows[view][method] = [json.loads((base / f"ipc3_sseed{s}.json").read_text()) for s in SEEDS]
    for view in VIEWS:
        for method in ("random_regions", "fg_regions"):
            for candidate, reference, seed in zip(rows[view][method], rows[view]["r0"], SEEDS):
                assert candidate["student_seed"] == reference["student_seed"] == seed
                assert candidate["initial_model_sha256"] == reference["initial_model_sha256"]
                assert candidate["fkd_hard_label"] and candidate["mix_type"] == "cutmix"
                assert candidate["epochs"] == 400 and candidate["temperature"] == 20
    groups = {view: {method: {metric: stats([x[metric] for x in rows[view][method]])
                                      for metric in METRICS} for method in METHODS} for view in VIEWS}
    contrasts = {}
    for view in VIEWS:
        for left, right in (("random_regions", "r0"), ("fg_regions", "r0"), ("fg_regions", "random_regions")):
            contrasts[f"{view}: {left} minus {right}"] = {
                metric: stats([a[metric] - b[metric] for a, b in zip(rows[view][left], rows[view][right])])
                for metric in METRICS}
    for method in METHODS:
        contrasts[f"{method}: rrc_cutmix minus fullframe_cutmix"] = {
            metric: stats([a[metric] - b[metric] for a, b in zip(rows["rrc_cutmix"][method], rows["fullframe_cutmix"][method])])
            for metric in METRICS}
    contrasts["RRC interaction: (fg minus random)_rrc minus (fg minus random)_fullframe"] = {
        metric: stats([(fg_on[metric] - random_on[metric]) - (fg_off[metric] - random_off[metric])
                       for fg_on, random_on, fg_off, random_off in zip(
                           rows["rrc_cutmix"]["fg_regions"], rows["rrc_cutmix"]["random_regions"],
                           rows["fullframe_cutmix"]["fg_regions"], rows["fullframe_cutmix"]["random_regions"])])
        for metric in METRICS}
    result = {"status": "complete", "protocol": "controlled_deco_style_regions_hard_rrc_factorial_v1",
              "groups": groups, "paired_contrasts": contrasts, "student_seeds": list(SEEDS),
              "new_fkd_sets": 0, "new_student_runs": 12}
    output = args.root / "summary/summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
