"""Summarize one-full-plus-two-mosaic Aircraft Soft results."""
import argparse
import json
import statistics
from pathlib import Path


SEEDS = (42, 43, 44)
GROUPS = ("r0", "hybrid_fg_uniform_rrc", "hybrid_fg_typed", "hybrid_random_typed")
METRICS = ("best_top1", "final_epoch_top1")


def stats(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "values": values}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--r0-results", required=True, type=Path)
    a = p.parse_args()
    rows = {name: [json.loads(((a.r0_results if name == "r0" else a.root / f"results/{name}") /
                              f"ipc3_sseed{s}.json").read_text()) for s in SEEDS] for name in GROUPS}
    for name in GROUPS[1:]:
        for candidate, reference, seed in zip(rows[name], rows["r0"], SEEDS):
            assert candidate["student_seed"] == reference["student_seed"] == seed
            assert candidate["initial_model_sha256"] == reference["initial_model_sha256"]
            assert candidate["epochs"] == 400 and candidate["temperature"] == 20
            assert candidate["mix_type"] == "cutmix" and not candidate["fkd_hard_label"]
    groups = {name: {metric: stats([row[metric] for row in rows[name]]) for metric in METRICS} for name in GROUPS}
    pairs = (("hybrid_fg_typed", "hybrid_fg_uniform_rrc"),
             ("hybrid_fg_typed", "hybrid_random_typed"),
             ("hybrid_fg_uniform_rrc", "r0"), ("hybrid_fg_typed", "r0"),
             ("hybrid_random_typed", "r0"))
    contrasts = {f"{left} minus {right}": {
        metric: stats([x[metric] - y[metric] for x, y in zip(rows[left], rows[right])]) for metric in METRICS}
        for left, right in pairs}
    construction = json.loads((a.root / "construction/construction_manifest.json").read_text())
    result = {"status": "complete", "protocol": "deco_hybrid_one_full_two_mosaics_soft_v2",
              "groups": groups, "paired_contrasts": contrasts, "student_seeds": list(SEEDS),
              "storage": {"ipc": 3, "full_slots": 1, "mosaic_slots": 2,
                          "independent_sources_per_class": construction["independent_sources_per_class"]},
              "new_fkd_sets": 3, "new_student_runs": 9}
    out = a.root / "summary/summary.json"; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
