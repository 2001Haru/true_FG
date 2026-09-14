"""Summarize Aircraft ViT-family teachers and their R0 downstream students."""

import argparse
import json
import statistics
from pathlib import Path

from vit_teacher_common import atomic_json


def aggregate(paths: list[Path]) -> dict:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    best = [float(row["best_top1"]) for row in rows]
    final = [float(row["final_epoch_top1"]) for row in rows]
    return {
        "runs": len(rows),
        "student_seeds": [int(row["student_seed"]) for row in rows],
        "best_mean": statistics.mean(best),
        "best_sample_sd": statistics.stdev(best),
        "final_mean": statistics.mean(final),
        "final_sample_sd": statistics.stdev(final),
        "best_values": best,
        "final_values": final,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--r0-result-root", required=True, type=Path)
    args = parser.parse_args()
    payload = {"protocol": "aircraft_R0_v2_labeler_comparison_v1", "teachers": {}, "students": {}}
    for kind in ("vit", "transfg"):
        teacher_manifest = args.root / "teachers" / kind / "training_manifest.json"
        fkd_manifest = args.root / "fkd" / kind / "ipc3_bs20_ipc3_fp32" / "relabel_manifest.json"
        payload["teachers"][kind] = json.loads(teacher_manifest.read_text(encoding="utf-8"))
        payload["teachers"][kind]["fkd"] = json.loads(fkd_manifest.read_text(encoding="utf-8"))
        paths = [args.root / "results" / kind / f"ipc3_sseed{seed}.json" for seed in (42, 43, 44)]
        payload["students"][kind] = aggregate(paths)
    baseline_paths = [
        args.r0_result_root / f"ipc3_sseed{seed}.json" for seed in (42, 43, 44)
    ]
    payload["students"]["resnet18_teacher42_existing"] = aggregate(baseline_paths)
    baseline = payload["students"]["resnet18_teacher42_existing"]
    for kind in ("vit", "transfg"):
        payload["students"][kind]["delta_vs_resnet_teacher"] = {
            "best": payload["students"][kind]["best_mean"] - baseline["best_mean"],
            "final": payload["students"][kind]["final_mean"] - baseline["final_mean"],
        }
    payload["students"]["transfg"]["delta_vs_plain_vit"] = {
        "best": payload["students"]["transfg"]["best_mean"] - payload["students"]["vit"]["best_mean"],
        "final": payload["students"]["transfg"]["final_mean"] - payload["students"]["vit"]["final_mean"],
    }
    atomic_json(payload, args.root / "summary" / "summary.json")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
