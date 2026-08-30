import argparse
import json
from pathlib import Path

from audit_result import audit_payload
from config import DATASETS
from summarize_locked_protocol import (
    DATASET_ORDER,
    IPCS,
    VALIDATION_IMAGES,
    final_top1_from_log,
    finite_top1,
    target_stats,
)


DESIGN = (
    (41, 42),
    (41, 43),
    (41, 44),
    (42, 42),
    (43, 42),
)


def metric_text(summary):
    if summary["count"] == 0:
        return "—"
    text = f"{summary['mean']:.2f}"
    if summary["sample_std"] is not None:
        text += f" ± {summary['sample_std']:.2f}"
    return text + f" ({summary['mean_minus_target']:+.2f})"


def main():
    parser = argparse.ArgumentParser("Summarize student- and recovery-seed variation")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("/linxi/dataset/FG_SRe2L_repro/v1/diagnostics/student_imagenet/results"),
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("/linxi/dataset/FG_SRe2L_repro/v1/diagnostics/student_imagenet/logs/jobs"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    runs = []
    missing = []
    for dataset_name in DATASET_ORDER:
        dataset = DATASETS[dataset_name]
        for ipc_index, ipc in enumerate(IPCS):
            for recovery_seed, student_seed in DESIGN:
                path = (
                    args.result_root / dataset_name / f"rseed{recovery_seed}"
                    / f"ipc{ipc}_sseed{student_seed}.json"
                )
                identity = {
                    "dataset": dataset_name,
                    "ipc": ipc,
                    "recovery_seed": recovery_seed,
                    "student_seed": student_seed,
                }
                if not path.is_file():
                    missing.append({**identity, "path": str(path)})
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                audit_payload(payload, dataset.classes, VALIDATION_IMAGES[dataset_name])
                if payload.get("student_initialization") != "imagenet-v1":
                    raise RuntimeError(f"Unexpected student initialization in {path}")
                best = finite_top1(payload["best_top1"], f"best_top1 in {path}")
                final = payload.get("final_epoch_top1")
                final_source = "result_json" if final is not None else None
                if final is None and recovery_seed == 41:
                    final, final_source = final_top1_from_log(
                        args.log_root, dataset_name, ipc, student_seed
                    )
                if final is not None:
                    final = finite_top1(final, f"final_epoch_top1 in {path}")
                runs.append({
                    **identity,
                    "best_top1": best,
                    "final_epoch_top1": final,
                    "final_epoch_source": final_source,
                    "target": dataset.paper_targets[ipc_index],
                    "path": str(path.resolve()),
                })

    groups = []
    for dataset_name in DATASET_ORDER:
        for ipc_index, ipc in enumerate(IPCS):
            target = DATASETS[dataset_name].paper_targets[ipc_index]
            group_runs = [
                run for run in runs
                if run["dataset"] == dataset_name and run["ipc"] == ipc
            ]
            student_runs = [run for run in group_runs if run["recovery_seed"] == 41]
            recovery_runs = [run for run in group_runs if run["student_seed"] == 42]

            def summarize(selected, key):
                return target_stats(
                    [run[key] for run in selected if run[key] is not None], target
                )

            groups.append({
                "dataset": dataset_name,
                "ipc": ipc,
                "target": target,
                "student_variation": {
                    "design": "rseed41 × sseed42/43/44",
                    "best": summarize(student_runs, "best_top1"),
                    "final": summarize(student_runs, "final_epoch_top1"),
                },
                "recovery_variation": {
                    "design": "rseed41/42/43 × sseed42",
                    "best": summarize(recovery_runs, "best_top1"),
                    "final": summarize(recovery_runs, "final_epoch_top1"),
                },
                "combined_unique": {
                    "design": "five unique crossed observations",
                    "best": summarize(group_runs, "best_top1"),
                    "final": summarize(group_runs, "final_epoch_top1"),
                },
            })

    expected_runs = len(DATASET_ORDER) * len(IPCS) * len(DESIGN)
    payload = {
        "status": "complete" if len(runs) == expected_runs else "incomplete",
        "protocol": "historical intended ImageNet-V1 teacher + ImageNet-V1 student",
        "design": [
            {"recovery_seed": recovery_seed, "student_seed": student_seed}
            for recovery_seed, student_seed in DESIGN
        ],
        "expected_runs": expected_runs,
        "completed_runs": len(runs),
        "missing": missing,
        "groups": groups,
        "runs": runs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "seed_variance_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Locked SRe2L++ seed-variance results",
        "",
        "All parenthesized values are mean minus the FD2 target.",
        "",
        "| Dataset | IPC | Target | Student variation best | Recovery variation best | Combined best | Combined final | Runs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in groups:
        lines.append(
            f"| {group['dataset']} | {group['ipc']} | {group['target']:.1f} | "
            f"{metric_text(group['student_variation']['best'])} | "
            f"{metric_text(group['recovery_variation']['best'])} | "
            f"{metric_text(group['combined_unique']['best'])} | "
            f"{metric_text(group['combined_unique']['final'])} | "
            f"{group['combined_unique']['best']['count']}/5 |"
        )
    (args.output_dir / "seed_variance_results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": payload["status"],
        "completed_runs": len(runs),
        "expected_runs": expected_runs,
    }))


if __name__ == "__main__":
    main()
