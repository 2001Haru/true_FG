"""Summarize the Hard-v1 RRC extension for Aircraft IPC3 DINO selections."""
import argparse
import json
import statistics
from pathlib import Path


METRICS = ("best_top1_diagnostic", "final_top1")


def stats(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "values": values}


def load(root, arm, seeds):
    return [json.loads((root / arm / f"sseed{s}.json").read_text()) for s in seeds]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--r0-root", required=True, type=Path)
    a = p.parse_args()
    mild_root, rrc_root = a.root / "results", a.root / "results_rrc"
    mild_top = {r["student_seed"]: r for r in load(mild_root, "global_center_top3", (42, 43, 44))}
    rrc_top = {r["student_seed"]: r for r in load(rrc_root, "global_center_top3", (42, 43, 44))}
    mild_k = {(rs, ss): json.loads((mild_root / f"spherical_kmeans3_rseed{rs}/sseed{ss}.json").read_text())
              for rs in (0, 1) for ss in (42, 43, 44)}
    rrc_k = {(rs, ss): json.loads((rrc_root / f"spherical_kmeans3_rseed{rs}/sseed{ss}.json").read_text())
             for rs in (0, 1) for ss in (42, 43, 44)}
    r0 = {mode: {r["student_seed"]: r for r in load(a.r0_root / mode, "r0", (42, 43, 44))}
          for mode in ("mild_rc", "rrc")}
    for row in list(rrc_top.values()) + list(rrc_k.values()):
        assert row["protocol"] == "hard_label_v1_rrc" and row["train_crop_mode"] == "rrc"
        assert row["updates_completed"] == 3000 and not row["cutmix"] and not row["mixup"]
    groups = {
        "rrc_global_center_top3": {m: stats([r[m] for r in rrc_top.values()]) for m in METRICS},
        "rrc_spherical_kmeans3": {m: stats([r[m] for r in rrc_k.values()]) for m in METRICS},
        "mild_global_center_top3_matched": {m: stats([r[m] for r in mild_top.values()]) for m in METRICS},
        "mild_spherical_kmeans3_matched": {m: stats([r[m] for r in mild_k.values()]) for m in METRICS},
    }
    contrasts = {}
    contrasts["global_center_top3: rrc minus mild_rc"] = {
        m: stats([rrc_top[s][m] - mild_top[s][m] for s in (42, 43, 44)]) for m in METRICS}
    contrasts["spherical_kmeans3: rrc minus mild_rc"] = {
        m: stats([rrc_k[rs, ss][m] - mild_k[rs, ss][m] for rs in (0, 1) for ss in (42, 43, 44)]) for m in METRICS}
    for name, rows in (("global_center_top3", rrc_top),):
        contrasts[f"rrc {name} minus r0"] = {
            m: stats([rows[s][m] - r0["rrc"][s][m] for s in (42, 43, 44)]) for m in METRICS}
    mean_k = {s: {m: statistics.mean(rrc_k[rs, s][m] for rs in (0, 1)) for m in METRICS} for s in (42, 43, 44)}
    contrasts["rrc mean_kmeans3_selection minus r0"] = {
        m: stats([mean_k[s][m] - r0["rrc"][s][m] for s in (42, 43, 44)]) for m in METRICS}
    contrasts["rrc mean_kmeans3_selection minus global_center_top3"] = {
        m: stats([mean_k[s][m] - rrc_top[s][m] for s in (42, 43, 44)]) for m in METRICS}
    result = {"status": "complete", "protocol": "dino_ipc3_top3_kmeans3_hard_v1_rrc_extension",
              "groups": groups, "paired_contrasts": contrasts,
              "kmeans_selection_seeds": [0, 1], "student_seeds": [42, 43, 44],
              "new_student_runs": 9, "selection_reused": True}
    out = a.root / "summary/rrc_extension.json";out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n");print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
