import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def load_json(path: Path) -> dict:
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
    parser = argparse.ArgumentParser("Attach standard-protocol provenance to a result")
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--teacher-seed", required=True, type=int)
    parser.add_argument("--recovery-seed", required=True, type=int)
    parser.add_argument("--ipc", required=True, type=int)
    parser.add_argument("--student-seed", required=True, type=int)
    parser.add_argument("--teacher-dir", required=True, type=Path)
    parser.add_argument("--recovery-root", required=True, type=Path)
    parser.add_argument("--fkd-dir", required=True, type=Path)
    args = parser.parse_args()

    result = load_json(args.result)
    teacher_complete_path = args.teacher_dir / "complete.json"
    teacher_gate_path = args.teacher_dir / "teacher_gate.json"
    teacher_checkpoint_path = args.teacher_dir / "ResNet18.pth"
    recovery_manifest_path = args.recovery_root / "recovery_manifest.json"
    relabel_manifest_path = args.fkd_dir / "relabel_manifest.json"
    fkd_audit_path = args.fkd_dir / "fkd_audit.json"
    teacher = load_json(teacher_complete_path)
    teacher_gate = load_json(teacher_gate_path)
    recovery = load_json(recovery_manifest_path)
    relabel = load_json(relabel_manifest_path)
    fkd_audit = load_json(fkd_audit_path)

    expected_batch = 14 if args.dataset == "SC_imsize224" else 20
    expect(teacher["status"], "complete", "Teacher status")
    expect(teacher["seed"], args.teacher_seed, "Teacher seed")
    expect(teacher["initialization"], "imagenet-v1", "Teacher initialization")
    expect(teacher["dataloader_workers"], 8, "Teacher workers")
    expect(teacher["persistent_workers"], False, "Teacher persistent workers")
    expect(recovery["status"], "complete", "Recovery status")
    expect(recovery["recovery_seed"], args.recovery_seed, "Recovery seed")
    expect(recovery["teacher_sha256"], sha256(teacher_checkpoint_path), "Recovery Teacher hash")
    expect(relabel["status"], "complete", "Relabel status")
    expect(relabel["ipc"], args.ipc, "Relabel IPC")
    expect(relabel["batch_size"], expected_batch, "Relabel batch size")
    expect(relabel["workers"], 8, "Relabel workers")
    expect(relabel["persistent_workers"], False, "Relabel persistent workers")
    expect(relabel["teacher_mode"], "train", "Relabel Teacher mode")
    expect(relabel["epochs"], 400, "Relabel epochs")
    expect(relabel["temperature"], 20.0, "Relabel temperature")
    expect(relabel["mix_type"], "cutmix", "Relabel mix type")
    expect(fkd_audit["status"], "complete", "FKD audit status")
    expect(result["student_initialization"], "random", "Student initialization")
    expect(result["student_seed"], args.student_seed, "Student seed")
    expect(result["optimizer"], "adamw", "Student optimizer")
    expect(result["learning_rate"], 1e-3, "Student learning rate")
    expect(result["weight_decay"], 1e-5, "Student weight decay")
    expect(result["cosine_eta"], 2.0, "Student cosine eta")
    expect(result["epochs"], 400, "Student epochs")
    expect(result["batch_size"], expected_batch, "Student batch size")
    expect(result["gradient_accumulation_steps"], 2, "Student accumulation")
    expect(result["temperature"], 20.0, "Student temperature")
    expect(result["dataloader_workers"], 8, "Student workers")
    expect(result["persistent_workers"], True, "Student persistent workers")

    repo_root = Path(__file__).resolve().parents[2]
    git_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    result["standard_protocol"] = {
        "name": "standard_protocol",
        "version": "v1",
        "git_revision": git_revision,
        "dataset": args.dataset,
        "teacher_seed": args.teacher_seed,
        "recovery_seed": args.recovery_seed,
        "ipc": args.ipc,
        "student_seed": args.student_seed,
        "teacher_complete": str(teacher_complete_path.resolve()),
        "teacher_complete_sha256": sha256(teacher_complete_path),
        "teacher_checkpoint": str(teacher_checkpoint_path.resolve()),
        "teacher_checkpoint_sha256": sha256(teacher_checkpoint_path),
        "teacher_gate": str(teacher_gate_path.resolve()),
        "teacher_gate_passed": bool(teacher_gate["passed"]),
        "recovery_manifest": str(recovery_manifest_path.resolve()),
        "recovery_manifest_sha256": sha256(recovery_manifest_path),
        "relabel_manifest": str(relabel_manifest_path.resolve()),
        "relabel_manifest_sha256": sha256(relabel_manifest_path),
        "fkd_audit": str(fkd_audit_path.resolve()),
        "fkd_audit_sha256": sha256(fkd_audit_path),
    }
    temporary = args.result.with_suffix(args.result.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.result)
    print(json.dumps({
        "status": "complete",
        "result": str(args.result.resolve()),
        "teacher_seed": args.teacher_seed,
        "recovery_seed": args.recovery_seed,
        "student_seed": args.student_seed,
        "ipc": args.ipc,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
