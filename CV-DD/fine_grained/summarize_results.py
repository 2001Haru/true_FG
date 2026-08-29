import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from config import DATASETS


RESULT_PATTERN = re.compile(r"ipc(?P<ipc>\d+)_sseed(?P<sseed>\d+)\.json$")
EXPECTED_RECOVERY_SEEDS = (41, 42, 43)
EXPECTED_STUDENT_SEEDS = (42, 43, 44)
EXPECTED_IPCS = (1, 3, 5)


def load_runs(root: Path) -> list[dict]:
    runs = []
    for dataset_name in DATASETS:
        dataset_root = root / "results" / dataset_name
        for path in sorted(dataset_root.glob("rseed*/ipc*_sseed*.json")):
            match = RESULT_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            recovery_seed = int(path.parent.name.removeprefix("rseed"))
            ipc = int(match.group("ipc"))
            student_seed = int(match.group("sseed"))
            payload = json.loads(path.read_text(encoding="utf-8"))
            runs.append({
                "dataset": dataset_name,
                "ipc": ipc,
                "recovery_seed": recovery_seed,
                "student_seed": student_seed,
                "top1": float(payload["best_top1"]),
                "path": str(path.resolve()),
                "training_target": payload.get("training_target"),
                "num_classes": payload.get("num_classes"),
                "validation_images": payload.get("validation_images"),
            })
    return runs


def main() -> None:
    parser = argparse.ArgumentParser("Summarize fine-grained SRe2L++ reproduction results")
    parser.add_argument("--experiment-root", type=Path,
                        default=Path("/linxi/dataset/FG_SRe2L_repro/v1"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or args.experiment_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.experiment_root)
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["dataset"], run["ipc"])].append(run)

    rows = []
    missing = []
    for dataset_name, cfg in DATASETS.items():
        for target_index, ipc in enumerate(EXPECTED_IPCS):
            group = grouped[(dataset_name, ipc)]
            observed = {(run["recovery_seed"], run["student_seed"]) for run in group}
            expected = {
                (recovery_seed, student_seed)
                for recovery_seed in EXPECTED_RECOVERY_SEEDS
                for student_seed in EXPECTED_STUDENT_SEEDS
            }
            missing_group = sorted(expected - observed)
            if missing_group:
                missing.append({"dataset": dataset_name, "ipc": ipc, "runs": missing_group})
            values = [run["top1"] for run in group]
            target = cfg.paper_targets[target_index]
            rows.append({
                "dataset": dataset_name,
                "ipc": ipc,
                "count": len(values),
                "mean": statistics.mean(values) if values else None,
                "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                "minimum": min(values) if values else None,
                "maximum": max(values) if values else None,
                "fd2_target": target,
                "mean_minus_target": statistics.mean(values) - target if values else None,
                "missing": missing_group,
            })

    payload = {
        "experiment_root": str(args.experiment_root.resolve()),
        "expected_recovery_seeds": EXPECTED_RECOVERY_SEEDS,
        "expected_student_seeds": EXPECTED_STUDENT_SEEDS,
        "runs": runs,
        "summary": rows,
        "missing": missing,
        "complete": not missing,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Fine-grained SRe2L++ reproduction summary",
        "",
        "| Dataset | IPC | Runs | Mean | Std | Min | Max | FD2 target | Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        def value(name: str) -> str:
            item = row[name]
            return "-" if item is None else f"{item:.3f}"
        lines.append(
            f"| {row['dataset']} | {row['ipc']} | {row['count']}/9 | "
            f"{value('mean')} | {value('sample_std')} | {value('minimum')} | "
            f"{value('maximum')} | {row['fd2_target']:.3f} | {value('mean_minus_target')} |"
        )
    if missing:
        lines.extend(["", "## Missing runs", ""])
        for item in missing:
            formatted = ", ".join(f"r{r}/s{s}" for r, s in item["runs"])
            lines.append(f"- {item['dataset']} IPC{item['ipc']}: {formatted}")
    (output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(runs), "complete": not missing, "output": str(output_dir)}))
    if missing and not args.allow_incomplete:
        raise RuntimeError(f"Missing {sum(len(item['runs']) for item in missing)} expected runs")


if __name__ == "__main__":
    main()
