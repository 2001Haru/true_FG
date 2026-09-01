import argparse
import hashlib
import json
import os
from pathlib import Path

from audit_result import audit_payload


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expect(actual, expected, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def main() -> None:
    parser = argparse.ArgumentParser("Attach CUB-4k sensitivity provenance")
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--teacher-seed", required=True, type=int)
    parser.add_argument("--recovery-seed", required=True, type=int)
    parser.add_argument("--ipc", required=True, type=int)
    parser.add_argument("--student-seed", required=True, type=int)
    parser.add_argument("--teacher-dir", required=True, type=Path)
    parser.add_argument("--recovery-root", required=True, type=Path)
    parser.add_argument("--fkd-dir", required=True, type=Path)
    parser.add_argument("--standard-root", required=True, type=Path)
    parser.add_argument("--reused-source", type=Path)
    args = parser.parse_args()

    result = load(args.result)
    audit_payload(result, 200, 5794)
    teacher_path = args.teacher_dir / "ResNet18.pth"
    teacher_complete_path = args.teacher_dir / "complete.json"
    recovery_manifest_path = args.recovery_root / "recovery_manifest.json"
    recovery_audit_path = args.recovery_root / "recovery_output_audit.json"
    relabel_manifest_path = args.fkd_dir / "relabel_manifest.json"
    fkd_audit_path = args.fkd_dir / "fkd_audit.json"
    teacher = load(teacher_complete_path)
    recovery = load(recovery_manifest_path)
    recovery_audit = load(recovery_audit_path)
    relabel = load(relabel_manifest_path)
    fkd = load(fkd_audit_path)
    teacher_hash = sha256(teacher_path)

    expect(teacher["status"], "complete", "Teacher status")
    expect(teacher["seed"], args.teacher_seed, "Teacher seed")
    expect(teacher["initialization"], "imagenet-v1", "Teacher initialization")
    expect(recovery["status"], "complete", "Recovery status")
    expect(recovery["recovery_seed"], args.recovery_seed, "Recovery seed")
    expect(recovery["teacher_sha256"], teacher_hash, "Recovery Teacher hash")
    expect(recovery["protocol"]["iterations"], 4000, "Recovery iterations")
    expect(recovery_audit["status"], "complete", "Recovery audit")
    expect(relabel["status"], "complete", "Relabel status")
    expect(relabel["ipc"], args.ipc, "Relabel IPC")
    expect(relabel["teacher_sha256"], teacher_hash, "Relabel Teacher hash")
    expect(relabel["workers"], 8, "Relabel workers")
    expect(relabel["persistent_workers"], False, "Relabel persistent")
    expect(fkd["status"], "complete", "FKD audit")
    expect(fkd["images"], 200 * args.ipc, "FKD images")
    expect(result["student_initialization"], "random", "Student initialization")
    expect(result["student_seed"], args.student_seed, "Student seed")
    expect(result["epochs"], 400, "Student epochs")
    expect(result["batch_size"], 20, "Student batch")
    expect(result["gradient_accumulation_steps"], 2, "Student accumulation")
    expect(result["temperature"], 20.0, "Student temperature")
    expect(result["optimizer"], "adamw", "Student optimizer")
    expect(result["learning_rate"], 1e-3, "Student LR")
    expect(result["weight_decay"], 1e-5, "Student weight decay")
    expect(result["cosine_eta"], 2.0, "Student eta")
    expect(result["dataloader_workers"], 8, "Student workers")
    expect(result["persistent_workers"], True, "Student persistent")

    baseline = (args.standard_root / "results" / f"tseed{args.teacher_seed}" /
                "CUB_imsize224" / f"rseed{args.recovery_seed}" /
                f"ipc{args.ipc}_sseed{args.student_seed}.json")
    if not baseline.is_file():
        raise FileNotFoundError(baseline)
    result["cub4k_protocol"] = {
        "name": "cub_recovery_4000_full_matrix",
        "version": "v1",
        "teacher_seed": args.teacher_seed,
        "recovery_seed": args.recovery_seed,
        "ipc": args.ipc,
        "student_seed": args.student_seed,
        "recovery_iterations": 4000,
        "paired_baseline_iterations": 10000,
        "teacher_checkpoint": str(teacher_path.resolve()),
        "teacher_checkpoint_sha256": teacher_hash,
        "recovery_manifest": str(recovery_manifest_path.resolve()),
        "recovery_manifest_sha256": sha256(recovery_manifest_path),
        "recovery_output_audit": str(recovery_audit_path.resolve()),
        "relabel_manifest": str(relabel_manifest_path.resolve()),
        "relabel_manifest_sha256": sha256(relabel_manifest_path),
        "fkd_audit": str(fkd_audit_path.resolve()),
        "fkd_audit_sha256": sha256(fkd_audit_path),
        "paired_10k_result": str(baseline.resolve()),
        "paired_10k_result_sha256": sha256(baseline),
        "reused_source": str(args.reused_source.resolve()) if args.reused_source else None,
    }
    temporary = args.result.with_suffix(args.result.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.result)
    print(json.dumps({"status": "complete", "result": str(args.result.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
