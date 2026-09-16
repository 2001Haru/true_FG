"""Summarize the R0/BBox-retarget light-square-RRC paired experiment."""

import argparse
import json
import statistics
from pathlib import Path


SEEDS = (42, 43, 44)


def stats(values):
    values = list(map(float, values))
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "values": values}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--fullframe-root", required=True, type=Path)
    p.add_argument("--compression-root", required=True, type=Path)
    a = p.parse_args()
    new, old, errors = {}, {}, []
    for method in ("r0", "bbox_retarget112"):
        for seed in SEEDS:
            new_path = a.root / f"results/{method}/ipc3_sseed{seed}.json"
            old_path = (a.fullframe_root / f"ipc3_sseed{seed}.json" if method == "r0" else
                        a.compression_root / f"results/bbox_retarget112/ipc3_sseed{seed}.json")
            try:
                new[(method, seed)] = json.loads(new_path.read_text())
                old[(method, seed)] = json.loads(old_path.read_text())
                assert new[(method, seed)]["student_seed"] == seed
                assert new[(method, seed)]["training_target"] == "fkd_soft_label"
                assert new[(method, seed)]["total_optimizer_updates"] == 6000
                assert new[(method, seed)]["initial_model_sha256"] == old[(method, seed)]["initial_model_sha256"]
            except Exception as error:
                errors.append({"path": str(new_path), "error": repr(error)})
    groups = {}
    for method in ("r0", "bbox_retarget112"):
        groups[method] = {
            "light_rrc_best": stats(new[(method, s)]["best_top1"] for s in SEEDS),
            "light_rrc_final": stats(new[(method, s)]["final_epoch_top1"] for s in SEEDS),
            "fullframe_best": stats(old[(method, s)]["best_top1"] for s in SEEDS),
            "fullframe_final": stats(old[(method, s)]["final_epoch_top1"] for s in SEEDS),
            "paired_light_rrc_minus_fullframe_best": stats(new[(method, s)]["best_top1"] - old[(method, s)]["best_top1"] for s in SEEDS),
            "paired_light_rrc_minus_fullframe_final": stats(new[(method, s)]["final_epoch_top1"] - old[(method, s)]["final_epoch_top1"] for s in SEEDS),
        }
    result = {
        "status": "complete" if len(new) == 6 and not errors else "failed",
        "protocol": "aircraft_ipc3_retarget_light_square_rrc_soft_v2_v1",
        "groups": groups,
        "bbox_minus_r0_under_light_rrc": {
            "best": stats(new[("bbox_retarget112", s)]["best_top1"] - new[("r0", s)]["best_top1"] for s in SEEDS),
            "final": stats(new[("bbox_retarget112", s)]["final_epoch_top1"] - new[("r0", s)]["final_epoch_top1"] for s in SEEDS),
        },
        "gap_change_vs_fullframe": {
            "best": stats((new[("bbox_retarget112", s)]["best_top1"] - new[("r0", s)]["best_top1"]) -
                          (old[("bbox_retarget112", s)]["best_top1"] - old[("r0", s)]["best_top1"]) for s in SEEDS),
            "final": stats((new[("bbox_retarget112", s)]["final_epoch_top1"] - new[("r0", s)]["final_epoch_top1"]) -
                           (old[("bbox_retarget112", s)]["final_epoch_top1"] - old[("r0", s)]["final_epoch_top1"]) for s in SEEDS),
        },
        "trajectory_audit": json.loads((a.root / "audits/light_rrc_trajectory.json").read_text()),
        "new_fkd_sets": 2, "new_student_runs": 6, "errors": errors,
    }
    output = a.root / "summary/summary.json"; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "complete": raise SystemExit(1)


if __name__ == "__main__": main()
