import argparse
import json
import math
from pathlib import Path

from config import DATASETS


ARMS = (
    {
        "dataset": "A_imsize224",
        "label": "inferred teacher + random student, T20",
        "path": "results/A_imsize224/rseed41/ipc1_sseed42.json",
    },
    {
        "dataset": "A_imsize224",
        "label": "intended teacher + random student, T20",
        "path": "diagnostics/historical_intended/pipeline/results/A_imsize224/rseed41/ipc1_sseed42.json",
    },
    {
        "dataset": "A_imsize224",
        "label": "intended teacher + random student, T3",
        "path": "diagnostics/temperature_3/results/A_imsize224/rseed41/ipc1_sseed42.json",
    },
    {
        "dataset": "A_imsize224",
        "label": "intended teacher + random student, T1",
        "path": "diagnostics/temperature_1/results/A_imsize224/rseed41/ipc1_sseed42.json",
    },
    {
        "dataset": "A_imsize224",
        "label": "intended teacher + random student, hard label",
        "path": "diagnostics/hard_label/results/A_imsize224/rseed41/ipc1_sseed42.json",
    },
    {
        "dataset": "A_imsize224",
        "label": "intended teacher + ImageNet student with reset BN, T20",
        "path": "diagnostics/student_imagenet_reset_bn/results/A_imsize224/rseed41/ipc1_sseed42.json",
    },
    {
        "dataset": "A_imsize224",
        "label": "intended teacher + ImageNet student, T20",
        "path": "diagnostics/student_imagenet/results/A_imsize224/rseed41/ipc1_sseed42.json",
    },
    {
        "dataset": "CUB_imsize224",
        "label": "code-faithful random teacher + random student, T20",
        "path": "diagnostics/historical_plain/pipeline/results/CUB_imsize224/rseed41/ipc1_sseed42.json",
    },
    {
        "dataset": "CUB_imsize224",
        "label": "intended teacher + random student, T20",
        "path": "diagnostics/historical_intended/pipeline/results/CUB_imsize224/rseed41/ipc1_sseed42.json",
    },
    {
        "dataset": "CUB_imsize224",
        "label": "intended teacher + ImageNet student, T20",
        "path": "diagnostics/student_imagenet/results/CUB_imsize224/rseed41/ipc1_sseed42.json",
    },
    {
        "dataset": "SC_imsize224",
        "label": "intended teacher + random student, T20",
        "path": "diagnostics/historical_intended/pipeline/results/SC_imsize224/rseed41/ipc1_sseed42.json",
    },
    {
        "dataset": "SC_imsize224",
        "label": "intended teacher + ImageNet student, T20",
        "path": "diagnostics/student_imagenet/results/SC_imsize224/rseed41/ipc1_sseed42.json",
    },
)


COMPARISONS = (
    {
        "dataset": "A_imsize224",
        "intervention": "replace inferred teacher with intended ImageNet teacher",
        "baseline": "inferred teacher + random student, T20",
        "candidate": "intended teacher + random student, T20",
    },
    {
        "dataset": "CUB_imsize224",
        "intervention": "enable intended ImageNet teacher initialization",
        "baseline": "code-faithful random teacher + random student, T20",
        "candidate": "intended teacher + random student, T20",
    },
    {
        "dataset": "A_imsize224",
        "intervention": "load ImageNet convolutional weights but reset BN state",
        "baseline": "intended teacher + random student, T20",
        "candidate": "intended teacher + ImageNet student with reset BN, T20",
    },
    {
        "dataset": "A_imsize224",
        "intervention": "retain ImageNet BN state in addition to convolutional weights",
        "baseline": "intended teacher + ImageNet student with reset BN, T20",
        "candidate": "intended teacher + ImageNet student, T20",
    },
    {
        "dataset": "A_imsize224",
        "intervention": "replace random student with ImageNet student",
        "baseline": "intended teacher + random student, T20",
        "candidate": "intended teacher + ImageNet student, T20",
    },
    {
        "dataset": "CUB_imsize224",
        "intervention": "replace random student with ImageNet student",
        "baseline": "intended teacher + random student, T20",
        "candidate": "intended teacher + ImageNet student, T20",
    },
    {
        "dataset": "SC_imsize224",
        "intervention": "replace random student with ImageNet student",
        "baseline": "intended teacher + random student, T20",
        "candidate": "intended teacher + ImageNet student, T20",
    },
)


def main():
    parser = argparse.ArgumentParser("Summarize locked protocol diagnostics")
    parser.add_argument(
        "--base-root",
        type=Path,
        default=Path("/linxi/dataset/FG_SRe2L_repro/v1"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    arms = []
    by_identity = {}
    for spec in ARMS:
        path = args.base_root / spec["path"]
        if not path.is_file():
            raise RuntimeError(f"missing protocol diagnostic: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        best = float(result["best_top1"])
        if not math.isfinite(best) or not 0.0 <= best <= 100.0:
            raise RuntimeError(f"invalid best_top1 in {path}: {best}")
        if result.get("num_classes") != DATASETS[spec["dataset"]].classes:
            raise RuntimeError(f"class count mismatch in {path}")
        record = {
            **spec,
            "best_top1": best,
            "training_target": result.get("training_target"),
            "student_initialization": result.get("student_initialization", "legacy-unrecorded"),
            "result": str(path.resolve()),
        }
        arms.append(record)
        by_identity[(spec["dataset"], spec["label"])] = record

    comparisons = []
    for spec in COMPARISONS:
        baseline = by_identity[(spec["dataset"], spec["baseline"])]
        candidate = by_identity[(spec["dataset"], spec["candidate"])]
        comparisons.append({
            **spec,
            "baseline_top1": baseline["best_top1"],
            "candidate_top1": candidate["best_top1"],
            "delta": candidate["best_top1"] - baseline["best_top1"],
        })

    payload = {
        "status": "complete",
        "scope": "IPC1, recovery seed 41, student seed 42 single-variable diagnostics",
        "arms": arms,
        "comparisons": comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "protocol_diagnostics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# SRe2L++ protocol diagnostics",
        "",
        "All rows use IPC1, recovery seed 41, and student seed 42.",
        "",
        "## Arms",
        "",
        "| Dataset | Arm | Best Top-1 |",
        "| --- | --- | ---: |",
    ]
    lines.extend(
        f"| {arm['dataset']} | {arm['label']} | {arm['best_top1']:.2f} |"
        for arm in arms
    )
    lines.extend([
        "",
        "## Single-variable comparisons",
        "",
        "| Dataset | Intervention | Baseline | Candidate | Delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    lines.extend(
        f"| {row['dataset']} | {row['intervention']} | {row['baseline_top1']:.2f} | "
        f"{row['candidate_top1']:.2f} | {row['delta']:+.2f} |"
        for row in comparisons
    )
    (args.output_dir / "protocol_diagnostics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "arms": len(arms), "comparisons": len(comparisons)}))


if __name__ == "__main__":
    main()
