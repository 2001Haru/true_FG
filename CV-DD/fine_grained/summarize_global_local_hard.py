"""Summarize paired Hard-label RRC on/off global-local inset results."""
import argparse
import json
import statistics
from pathlib import Path


def stats(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "values": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--soft-root", required=True, type=Path)
    args = parser.parse_args()
    references = {
        "fullframe_cutmix": Path("/linxi/dataset/FG_HardAugFactorial_v2/aircraft_ipc3_v1/results/fullframe_cutmix/random_real/source_seed0"),
        "rrc_cutmix": Path("/linxi/dataset/FG_HardReplay_standard/v2/aircraft_v1/results/random_real/source_seed0"),
    }
    rows = {}
    for arm in ("fullframe_cutmix", "rrc_cutmix"):
        rows[(arm, "r0")] = [json.loads((references[arm] / f"ipc3_sseed{seed}.json").read_text()) for seed in (42, 43, 44)]
        for method in ("lowres_detail", "native_detail"):
            rows[(arm, method)] = [json.loads((args.root / f"results/{arm}/{method}/ipc3_sseed{seed}.json").read_text()) for seed in (42, 43, 44)]
    metrics = ("best_top1", "final_epoch_top1")
    result = {"status": "complete", "protocol": "global_local_inset_hard_rrc_factorial_v1", "groups": {}, "paired_deltas": {}}
    for (arm, method), group in rows.items():
        result["groups"][f"{arm}/{method}"] = {metric: stats([row[metric] for row in group]) for metric in metrics}
    for arm in ("fullframe_cutmix", "rrc_cutmix"):
        for candidate, reference in (("lowres_detail", "r0"), ("native_detail", "r0"), ("native_detail", "lowres_detail")):
            result["paired_deltas"][f"{arm}: {candidate} minus {reference}"] = {
                metric: stats([left[metric] - right[metric] for left, right in zip(rows[(arm, candidate)], rows[(arm, reference)])])
                for metric in metrics
            }
        for method in ("lowres_detail", "native_detail"):
            for candidate, reference in zip(rows[(arm, method)], rows[(arm, "r0")]):
                assert candidate["student_seed"] == reference["student_seed"]
                assert candidate["initial_model_sha256"] == reference["initial_model_sha256"]
                assert candidate["fkd_hard_label"] and candidate["training_target"] == "fkd_replay_hard_cutmix_ce"
                assert candidate["mix_type"] == "cutmix" and candidate["hard_cutmix_lambda_source"] == "actual_bbox_area"
                assert candidate["epochs"] == 400 and candidate["batch_size"] == 20
    result["soft_context"] = json.loads((args.soft_root / "summary/summary.json").read_text())
    output = args.root / "summary/summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
