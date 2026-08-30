import argparse
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_text(summary):
    if summary["count"] == 0:
        return "—"
    text = f"{summary['mean']:.2f}"
    if summary["sample_std"] is not None:
        text += f" ± {summary['sample_std']:.2f}"
    return text


def main():
    parser = argparse.ArgumentParser("Build the locked SRe2L++ reproduction report")
    parser.add_argument("--summary-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary_dir = args.summary_dir
    output = args.output or summary_dir / "reproduction_report.md"

    paths = {
        "locked": summary_dir / "locked_results.json",
        "seeds": summary_dir / "seed_variance_results.json",
        "diagnostics": summary_dir / "protocol_diagnostics.json",
        "provenance": summary_dir / "protocol_provenance.json",
        "release": summary_dir / "fd2_release_inventory.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing {name} evidence: {path}")
    evidence = {name: read_json(path) for name, path in paths.items()}
    required_statuses = {
        name: evidence[name]["status"]
        for name in ("locked", "seeds", "diagnostics", "provenance", "release")
    }
    status = "complete" if all(value == "complete" for value in required_statuses.values()) else "incomplete"

    lines = [
        "# SRe²L++ fine-grained reproduction report",
        "",
        f"**Status: {status}.**",
        "",
        "This report targets the FD² ResNet18 SRe²L++ baseline on CUB-200-2011, "
        "FGVC-Aircraft, and Stanford Cars at IPC 1/3/5.",
        "",
        "## Locked protocol",
        "",
        "- Teacher: torchvision ResNet18 ImageNet-1K V1 initialization, historical deleted "
        "CV-DD schedule (batch 32, 100 epochs, SGD 0.01, cosine), final checkpoint.",
        "- Recovery: IPC5 first, 2×2 patch initialization, dataset-specific 10k/4k iterations, "
        "then byte-identical IPC1/3 subsets.",
        "- Relabel/evaluation: FKD T20, CutMix, AdamW 1e-3/1e-5, eta=2, 400 epochs.",
        "- Student: torchvision ResNet18 ImageNet-1K V1 initialization. This is an intentional "
        "protocol reconstruction; the released random-student path does not reproduce the paper scale.",
        "- Historical-source caveat: the deleted plain-teacher trainer/launcher blob IDs and the "
        "13-commit audit are retained in the frozen release audit and teacher manifests, but those "
        "two deleted Git objects are no longer present in the current object database. The current "
        "teacher checkpoint, gates, recovery trees, and FKD artifacts remain directly hash-verifiable.",
        "",
        "## Main rseed41 × three student seeds",
        "",
        "| Dataset | IPC | FD² target | Best mean ± sd | Final mean ± sd | Closest gap | Runs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in evidence["locked"]["groups"]:
        gaps = [
            value for value in (
                group["best"]["closest_absolute_gap"],
                group["final"]["closest_absolute_gap"],
            ) if value is not None
        ]
        closest = "—" if not gaps else f"{min(gaps):.2f}"
        lines.append(
            f"| {group['dataset']} | {group['ipc']} | {group['target']:.1f} | "
            f"{metric_text(group['best'])} | {metric_text(group['final'])} | "
            f"{closest} | {group['best']['count']}/3 |"
        )

    lines.extend([
        "",
        "## Student-seed versus recovery-seed variation",
        "",
        "| Dataset | IPC | Student variation best | Recovery variation best | Combined best | Runs |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for group in evidence["seeds"]["groups"]:
        lines.append(
            f"| {group['dataset']} | {group['ipc']} | "
            f"{metric_text(group['student_variation']['best'])} | "
            f"{metric_text(group['recovery_variation']['best'])} | "
            f"{metric_text(group['combined_unique']['best'])} | "
            f"{group['combined_unique']['best']['count']}/5 |"
        )

    lines.extend([
        "",
        "## Protocol divergence diagnostics",
        "",
        "| Dataset | Single-variable intervention | Baseline | Candidate | Delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for row in evidence["diagnostics"]["comparisons"]:
        lines.append(
            f"| {row['dataset']} | {row['intervention']} | {row['baseline_top1']:.2f} | "
            f"{row['candidate_top1']:.2f} | {row['delta']:+.2f} |"
        )

    release = evidence["release"]
    lines.extend([
        "",
        "## Official release availability",
        "",
        f"The audited FD² snapshot contains {release['tracked_files']} tracked files, "
        f"{len(release['model_or_teacher_artifacts'])} model/offline artifacts, and "
        f"{len(release['distilled_or_patch_images_outside_figures'])} distilled or patch images "
        "outside paper figures. The plain baseline launchers reference `recover.py` and "
        "`recover_FADRM.py`, but neither entrypoint is tracked.",
        "",
        "The later DeCO paper repeats the FD² SRe²L++ numbers under the original protocols; "
        "it explicitly attributes them to prior work rather than an independent reproduction: "
        "https://arxiv.org/abs/2608.25480.",
        "",
        "## Evidence status",
        "",
        "| Evidence | Status | File |",
        "| --- | --- | --- |",
    ])
    for name, path in paths.items():
        lines.append(f"| {name} | {required_statuses[name]} | `{path.resolve()}` |")
    if status != "complete":
        lines.extend([
            "",
            "The report remains incomplete because one or more required matrices or provenance "
            "audits have not finished. No final reproduction claim is made at this stage.",
        ])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "output": str(output.resolve()), "evidence": required_statuses}))


if __name__ == "__main__":
    main()
