import argparse
import hashlib
import json
from pathlib import Path

from config import DATASETS
from summarize_locked_protocol import DATASET_ORDER, IPCS


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(f"protocol provenance audit failed: {message}")


def file_record(path):
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main():
    parser = argparse.ArgumentParser("Audit locked SRe2L++ protocol provenance")
    parser.add_argument(
        "--base-root",
        type=Path,
        default=Path("/linxi/dataset/FG_SRe2L_repro/v1"),
    )
    parser.add_argument("--recovery-seeds", nargs="+", type=int, default=[41])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    intended_root = args.base_root / "diagnostics" / "historical_intended"
    pipeline_root = intended_root / "pipeline"
    missing = []
    incomplete = []
    datasets = []

    for dataset_name in DATASET_ORDER:
        dataset = DATASETS[dataset_name]
        teacher_dir = intended_root / "teacher" / dataset_name / "tseed42"
        teacher_path = teacher_dir / "ResNet18.pth"
        teacher_files = {
            name: teacher_dir / name
            for name in ("manifest.json", "complete.json", "teacher_gate.json")
        }
        require(teacher_path.is_file(), f"missing teacher: {teacher_path}")
        for path in teacher_files.values():
            require(path.is_file(), f"missing teacher provenance: {path}")

        teacher_sha = sha256(teacher_path)
        teacher_manifest = read_json(teacher_files["manifest.json"])
        teacher_complete = read_json(teacher_files["complete.json"])
        teacher_gate = read_json(teacher_files["teacher_gate.json"])
        require(teacher_complete.get("status") == "complete", f"teacher incomplete: {dataset_name}")
        require(teacher_gate.get("passed") is True, f"teacher gate failed: {dataset_name}")
        require(teacher_gate.get("teacher_sha256") == teacher_sha, f"teacher gate hash mismatch: {dataset_name}")
        require(teacher_manifest.get("initialization") == "imagenet-v1", f"teacher initialization mismatch: {dataset_name}")
        require(teacher_manifest.get("batch_size") == 32, f"teacher batch mismatch: {dataset_name}")
        require(teacher_manifest.get("epochs") == 100, f"teacher epoch mismatch: {dataset_name}")

        patch_manifest_path = (
            pipeline_root / "patches" / dataset_name / "tseed42_pseed42"
            / "2" / "patch_manifest.json"
        )
        require(patch_manifest_path.is_file(), f"missing patch manifest: {dataset_name}")
        patch_manifest = read_json(patch_manifest_path)
        require(patch_manifest.get("status") == "complete", f"patches incomplete: {dataset_name}")
        require(patch_manifest.get("teacher_sha256") == teacher_sha, f"patch teacher mismatch: {dataset_name}")
        require(patch_manifest.get("files") == dataset.classes * 5, f"patch count mismatch: {dataset_name}")

        recovery_records = []
        for recovery_seed in args.recovery_seeds:
            recovery_dir = pipeline_root / "recovery" / dataset_name / f"rseed{recovery_seed}"
            manifest_path = recovery_dir / "recovery_manifest.json"
            audit_path = recovery_dir / "recovery_output_audit.json"
            identity = {"dataset": dataset_name, "recovery_seed": recovery_seed}
            if not manifest_path.is_file() or not audit_path.is_file():
                missing.append({
                    **identity,
                    "stage": "recovery",
                    "paths": [str(manifest_path), str(audit_path)],
                })
                continue
            recovery_manifest = read_json(manifest_path)
            recovery_audit = read_json(audit_path)
            if recovery_manifest.get("status") != "complete" or recovery_audit.get("status") != "complete":
                incomplete.append({**identity, "stage": "recovery"})
                continue
            require(recovery_manifest.get("teacher_sha256") == teacher_sha, f"recovery teacher mismatch: {identity}")
            require(recovery_audit.get("teacher_sha256") == teacher_sha, f"recovery audit teacher mismatch: {identity}")
            require(
                recovery_audit.get("sampling_relation")
                == "IPC1 and IPC3 are byte-identical relative-path subsets of IPC5",
                f"sampling relation mismatch: {identity}",
            )
            for ipc in IPCS:
                tree = recovery_audit["trees"][str(ipc)]
                require(tree["files"] == dataset.classes * ipc, f"recovery file count mismatch: {identity}, IPC{ipc}")

            fkd_records = []
            for ipc in IPCS:
                fkd_audit_path = (
                    pipeline_root / "fkd" / dataset_name / f"rseed{recovery_seed}"
                    / f"ipc{ipc}_bs{dataset.fkd_batch_size}_ipc{ipc}" / "fkd_audit.json"
                )
                if not fkd_audit_path.is_file():
                    missing.append({**identity, "stage": "fkd", "ipc": ipc, "path": str(fkd_audit_path)})
                    continue
                fkd_audit = read_json(fkd_audit_path)
                expected_batches = 400 * dataset.classes * ipc // dataset.fkd_batch_size
                require(fkd_audit.get("status") == "complete", f"FKD incomplete: {identity}, IPC{ipc}")
                require(fkd_audit.get("batch_files") == expected_batches, f"FKD count mismatch: {identity}, IPC{ipc}")
                require(fkd_audit.get("classes") == dataset.classes, f"FKD class mismatch: {identity}, IPC{ipc}")
                fkd_records.append({"ipc": ipc, "audit": file_record(fkd_audit_path), "summary": fkd_audit})

            recovery_records.append({
                "recovery_seed": recovery_seed,
                "manifest": file_record(manifest_path),
                "audit": file_record(audit_path),
                "tree_hashes": {
                    ipc: recovery_audit["trees"][str(ipc)]["tree_sha256"] for ipc in IPCS
                },
                "fkd": fkd_records,
            })

        datasets.append({
            "dataset": dataset_name,
            "teacher": {
                "checkpoint": file_record(teacher_path),
                "validation_accuracy": teacher_gate["best_validation_accuracy"],
                "gate_reference": teacher_gate["reference_top1"],
                "gate_passed": True,
                "manifest_status": teacher_manifest.get("status"),
                "complete_marker": file_record(teacher_files["complete.json"]),
                "gate": file_record(teacher_files["teacher_gate.json"]),
                "protocol": teacher_manifest,
            },
            "patches": {
                "manifest": file_record(patch_manifest_path),
                "tree_sha256": None,
                "files": patch_manifest["files"],
            },
            "recoveries": recovery_records,
        })

    payload = {
        "status": "complete" if not missing and not incomplete else "incomplete",
        "base_root": str(args.base_root.resolve()),
        "requested_recovery_seeds": args.recovery_seeds,
        "missing": missing,
        "incomplete": incomplete,
        "datasets": datasets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "datasets": len(datasets),
        "missing": len(missing),
        "incomplete": len(incomplete),
        "output": str(args.output.resolve()),
    }))


if __name__ == "__main__":
    main()
