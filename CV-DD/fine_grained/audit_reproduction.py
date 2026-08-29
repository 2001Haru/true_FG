import argparse
import hashlib
import json
from pathlib import Path

from config import DATASETS
from summarize_results import (
    EXPECTED_IPCS,
    EXPECTED_RECOVERY_SEEDS,
    EXPECTED_STUDENT_SEEDS,
    EXPECTED_VALIDATION_IMAGES,
    load_runs,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, problems: list[str]) -> dict | None:
    if not path.is_file():
        problems.append(f"missing: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        problems.append(f"invalid JSON {path}: {error}")
        return None


def expect(condition: bool, message: str, problems: list[str]) -> None:
    if not condition:
        problems.append(message)


def main() -> None:
    parser = argparse.ArgumentParser("Audit the complete fine-grained SRe2L++ reproduction")
    parser.add_argument(
        "--experiment-root", type=Path,
        default=Path("/linxi/dataset/FG_SRe2L_repro/v1"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root = args.experiment_root
    problems: list[str] = []
    evidence = {"datasets": {}, "result_runs": 0}

    for dataset_name, cfg in DATASETS.items():
        dataset_evidence = {"recoveries": {}}
        evidence["datasets"][dataset_name] = dataset_evidence

        input_audit_path = root / "audits" / f"{dataset_name}.json"
        input_audit = read_json(input_audit_path, problems)
        if input_audit:
            expect(input_audit.get("status") == "valid", f"input audit not valid: {input_audit_path}", problems)
            expect(input_audit.get("verify_all_sizes") is True, f"input audit did not inspect all sizes: {input_audit_path}", problems)
            expect(input_audit.get("train", {}).get("classes") == input_audit.get("test", {}).get("classes"), f"train/test classes differ: {input_audit_path}", problems)
            expect(input_audit.get("test", {}).get("images") == EXPECTED_VALIDATION_IMAGES[dataset_name], f"test image count mismatch: {dataset_name}", problems)
            for split in ("train", "test"):
                split_data = input_audit.get(split, {})
                expect(split_data.get("inspected_sizes") == {"224x224": split_data.get("images")}, f"not every {split} image was audited at 224x224: {dataset_name}", problems)
            dataset_evidence["input_audit"] = str(input_audit_path.resolve())

        stats_path = root / "datasets" / dataset_name / "train_channel_stats.json"
        channel_stats = read_json(stats_path, problems)
        if channel_stats:
            expect(channel_stats.get("status") == "complete", f"channel statistics incomplete: {dataset_name}", problems)
            if input_audit:
                expect(channel_stats.get("images") == input_audit.get("train", {}).get("images"), f"channel-stat/input train count mismatch: {dataset_name}", problems)
            dataset_evidence["channel_statistics"] = {
                "path": str(stats_path.resolve()),
                "mean": channel_stats.get("mean"),
                "std_population": channel_stats.get("std_population"),
                "max_abs_mean_delta": channel_stats.get("max_abs_mean_delta"),
                "max_abs_std_delta": channel_stats.get("max_abs_std_delta"),
            }

        teacher_dir = root / "teachers" / dataset_name / "tseed42"
        teacher_checkpoint = teacher_dir / "ResNet18.pth"
        teacher_complete = read_json(teacher_dir / "complete.json", problems)
        teacher_gate = read_json(teacher_dir / "teacher_gate.json", problems)
        teacher_hash = sha256(teacher_checkpoint) if teacher_checkpoint.is_file() else None
        if teacher_complete:
            expect(teacher_complete.get("status") == "complete", f"teacher incomplete: {dataset_name}", problems)
            expect(teacher_complete.get("final_epoch") == 50, f"teacher final epoch is not 50: {dataset_name}", problems)
            if input_audit:
                expect(teacher_complete.get("train_images") == input_audit.get("train", {}).get("images"), f"teacher/input train count mismatch: {dataset_name}", problems)
                expect(teacher_complete.get("test_images") == input_audit.get("test", {}).get("images"), f"teacher/input test count mismatch: {dataset_name}", problems)
        if teacher_gate:
            expect(teacher_gate.get("passed") is True, f"teacher gate failed: {dataset_name}", problems)
            expect(teacher_gate.get("teacher_sha256") == teacher_hash, f"teacher hash/gate mismatch: {dataset_name}", problems)
        dataset_evidence["teacher"] = {
            "checkpoint": str(teacher_checkpoint.resolve()),
            "sha256": teacher_hash,
            "best_top1": teacher_complete.get("best_validation_accuracy") if teacher_complete else None,
        }

        patch_dir = root / "patches" / dataset_name / "tseed42_pseed42" / "2"
        patch_manifest = read_json(patch_dir / "patch_manifest.json", problems)
        if patch_manifest:
            expect(patch_manifest.get("status") == "complete", f"patch manifest incomplete: {dataset_name}", problems)
            expect(patch_manifest.get("files") == cfg.classes * 5, f"patch count mismatch: {dataset_name}", problems)
            expect(patch_manifest.get("teacher_sha256") == teacher_hash, f"patch teacher hash mismatch: {dataset_name}", problems)
        patch_hash = None

        tree_hashes_by_ipc = {ipc: [] for ipc in EXPECTED_IPCS}
        for recovery_seed in EXPECTED_RECOVERY_SEEDS:
            recovery_root = root / "recovery" / dataset_name / f"rseed{recovery_seed}"
            recovery_manifest = read_json(recovery_root / "recovery_manifest.json", problems)
            output_audit = read_json(recovery_root / "recovery_output_audit.json", problems)
            if recovery_manifest:
                expect(recovery_manifest.get("status") == "complete", f"recovery incomplete: {dataset_name} r{recovery_seed}", problems)
                expect(recovery_manifest.get("recovery_seed") == recovery_seed, f"recovery seed mismatch: {dataset_name} r{recovery_seed}", problems)
                expect(recovery_manifest.get("teacher_sha256") == teacher_hash, f"recovery teacher hash mismatch: {dataset_name} r{recovery_seed}", problems)
                if patch_hash is None:
                    patch_hash = recovery_manifest.get("patch_tree_sha256")
                expect(recovery_manifest.get("patch_tree_sha256") == patch_hash, f"recovery patch hash mismatch: {dataset_name} r{recovery_seed}", problems)
            if output_audit:
                expect(output_audit.get("status") == "complete", f"output audit incomplete: {dataset_name} r{recovery_seed}", problems)
                expect(output_audit.get("recovery_seed") == recovery_seed, f"output audit seed mismatch: {dataset_name} r{recovery_seed}", problems)
                expect(output_audit.get("teacher_sha256") == teacher_hash, f"output audit teacher mismatch: {dataset_name} r{recovery_seed}", problems)
                expect(output_audit.get("patch_tree_sha256") == patch_hash, f"output audit patch mismatch: {dataset_name} r{recovery_seed}", problems)
                for ipc in EXPECTED_IPCS:
                    tree = output_audit.get("trees", {}).get(str(ipc), {})
                    expect(tree.get("files") == cfg.classes * ipc, f"IPC{ipc} image count mismatch: {dataset_name} r{recovery_seed}", problems)
                    if tree.get("tree_sha256"):
                        tree_hashes_by_ipc[ipc].append(tree["tree_sha256"])

            fkd_audits = {}
            for ipc in EXPECTED_IPCS:
                fkd_dir = root / "fkd" / dataset_name / f"rseed{recovery_seed}" / f"ipc{ipc}_bs{cfg.fkd_batch_size}_ipc{ipc}"
                fkd_audit = read_json(fkd_dir / "fkd_audit.json", problems)
                if fkd_audit:
                    expect(fkd_audit.get("status") == "complete", f"FKD audit incomplete: {dataset_name} r{recovery_seed} IPC{ipc}", problems)
                    expect(fkd_audit.get("epochs") == 400, f"FKD epoch mismatch: {dataset_name} r{recovery_seed} IPC{ipc}", problems)
                    expect(fkd_audit.get("images") == cfg.classes * ipc, f"FKD image count mismatch: {dataset_name} r{recovery_seed} IPC{ipc}", problems)
                    fkd_audits[str(ipc)] = str((fkd_dir / "fkd_audit.json").resolve())
            dataset_evidence["recoveries"][str(recovery_seed)] = {
                "manifest": str((recovery_root / "recovery_manifest.json").resolve()),
                "output_audit": str((recovery_root / "recovery_output_audit.json").resolve()),
                "fkd_audits": fkd_audits,
            }

        for ipc, hashes in tree_hashes_by_ipc.items():
            if len(hashes) == len(EXPECTED_RECOVERY_SEEDS):
                expect(len(set(hashes)) == len(hashes), f"duplicate recovery trees: {dataset_name} IPC{ipc}", problems)

        diversity_path = root / "audits" / f"{dataset_name}_recovery_seed_diversity.json"
        diversity = read_json(diversity_path, problems)
        if diversity:
            expect(diversity.get("status") == "complete", f"recovery diversity audit incomplete: {dataset_name}", problems)
            expect(diversity.get("relative_paths") == cfg.classes * 5, f"recovery diversity path count mismatch: {dataset_name}", problems)
            pairs = diversity.get("pairs", [])
            expect(len(pairs) == 3, f"recovery diversity pair count mismatch: {dataset_name}", problems)
            for pair in pairs:
                expect(pair.get("exact_duplicates") == 0, f"exact cross-seed recovery duplicates: {dataset_name} {pair.get('left_seed')}/{pair.get('right_seed')}", problems)
                expect(pair.get("mean_mae_0_1", 0) > 0, f"zero cross-seed recovery MAE: {dataset_name}", problems)
            dataset_evidence["recovery_seed_diversity"] = str(diversity_path.resolve())

    try:
        runs = load_runs(root)
    except (KeyError, TypeError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as error:
        problems.append(f"result validation failed: {error}")
        runs = []
    evidence["result_runs"] = len(runs)
    expected_runs = {
        (dataset, ipc, recovery_seed, student_seed)
        for dataset in DATASETS
        for ipc in EXPECTED_IPCS
        for recovery_seed in EXPECTED_RECOVERY_SEEDS
        for student_seed in EXPECTED_STUDENT_SEEDS
    }
    observed_runs = {
        (run["dataset"], run["ipc"], run["recovery_seed"], run["student_seed"])
        for run in runs
    }
    for item in sorted(expected_runs - observed_runs):
        problems.append(f"missing result: {item}")
    for item in sorted(observed_runs - expected_runs):
        problems.append(f"unexpected result: {item}")

    payload = {
        "status": "complete" if not problems else "incomplete",
        "experiment_root": str(root.resolve()),
        "expected_result_runs": len(expected_runs),
        "evidence": evidence,
        "problems": problems,
    }
    output = args.output or root / "summary" / "completion_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "problems": len(problems),
        "result_runs": len(runs),
        "output": str(output),
    }, sort_keys=True))
    if problems and not args.allow_incomplete:
        raise RuntimeError(f"Reproduction audit found {len(problems)} problems")


if __name__ == "__main__":
    main()
