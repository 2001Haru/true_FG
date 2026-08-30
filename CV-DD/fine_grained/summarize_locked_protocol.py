import argparse
import json
import math
import re
import statistics
from pathlib import Path

from config import DATASETS


DATASET_ORDER = ("CUB_imsize224", "A_imsize224", "SC_imsize224")
IPCS = (1, 3, 5)
STUDENT_SEEDS = (42, 43, 44)
RECOVERY_SEED = 41
RESULT_RE = re.compile(r"ipc(?P<ipc>[135])_sseed(?P<seed>\d+)\.json$")


def finite_top1(value, label):
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 100.0:
        raise RuntimeError(f"Invalid {label}: {value}")
    return value


def stats(values):
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def main():
    parser = argparse.ArgumentParser("Summarize the locked intended SRe2L++ protocol")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("/linxi/dataset/FG_SRe2L_repro/v1/diagnostics/student_imagenet/results"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    runs = []
    missing = []
    for dataset_name in DATASET_ORDER:
        for ipc_index, ipc in enumerate(IPCS):
            for student_seed in STUDENT_SEEDS:
                path = (
                    args.result_root / dataset_name / f"rseed{RECOVERY_SEED}"
                    / f"ipc{ipc}_sseed{student_seed}.json"
                )
                if not path.is_file():
                    missing.append({
                        "dataset": dataset_name,
                        "ipc": ipc,
                        "student_seed": student_seed,
                        "path": str(path),
                    })
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("training_target") != "fkd_soft_label":
                    raise RuntimeError(f"Unexpected target in {path}")
                if payload.get("student_initialization") != "imagenet-v1":
                    raise RuntimeError(f"Unexpected student initialization in {path}")
                best = finite_top1(payload["best_top1"], f"best_top1 in {path}")
                final = payload.get("final_epoch_top1")
                if final is not None:
                    final = finite_top1(final, f"final_epoch_top1 in {path}")
                runs.append({
                    "dataset": dataset_name,
                    "ipc": ipc,
                    "recovery_seed": RECOVERY_SEED,
                    "student_seed": student_seed,
                    "best_top1": best,
                    "final_epoch_top1": final,
                    "target": DATASETS[dataset_name].paper_targets[ipc_index],
                    "path": str(path.resolve()),
                })

    groups = []
    for dataset_name in DATASET_ORDER:
        for ipc_index, ipc in enumerate(IPCS):
            group = [run for run in runs if run["dataset"] == dataset_name and run["ipc"] == ipc]
            best_values = [run["best_top1"] for run in group]
            final_values = [run["final_epoch_top1"] for run in group if run["final_epoch_top1"] is not None]
            target = DATASETS[dataset_name].paper_targets[ipc_index]
            groups.append({
                "dataset": dataset_name,
                "ipc": ipc,
                "target": target,
                "best": {**stats(best_values), "mean_minus_target": statistics.mean(best_values) - target if best_values else None},
                "final": {**stats(final_values), "mean_minus_target": statistics.mean(final_values) - target if final_values else None},
                "student_seeds": sorted(run["student_seed"] for run in group),
            })

    complete = len(runs) == len(DATASET_ORDER) * len(IPCS) * len(STUDENT_SEEDS)
    payload = {
        "status": "complete" if complete else "incomplete",
        "protocol": "historical intended ImageNet-V1 teacher + ImageNet-V1 student",
        "result_root": str(args.result_root.resolve()),
        "expected_runs": 27,
        "completed_runs": len(runs),
        "missing": missing,
        "groups": groups,
        "runs": runs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "locked_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Locked intended SRe2L++ results",
        "",
        "| Dataset | IPC | FD2 target | Best mean ± sd | Final mean ± sd | Runs |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in groups:
        best = group["best"]
        final = group["final"]
        best_text = "—" if best["count"] == 0 else f"{best['mean']:.2f} ± {best['sample_std']:.2f}" if best["sample_std"] is not None else f"{best['mean']:.2f}"
        final_text = "—" if final["count"] == 0 else f"{final['mean']:.2f} ± {final['sample_std']:.2f}" if final["sample_std"] is not None else f"{final['mean']:.2f}"
        lines.append(
            f"| {group['dataset']} | {group['ipc']} | {group['target']:.1f} | "
            f"{best_text} | {final_text} | {best['count']}/3 |"
        )
    (args.output_dir / "locked_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "completed_runs": len(runs), "expected_runs": 27}))


if __name__ == "__main__":
    main()
