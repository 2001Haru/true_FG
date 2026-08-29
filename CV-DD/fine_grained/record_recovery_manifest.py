import argparse
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

from config import get_dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*.jpg") if path.is_file())
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest(), len(files)


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser("Write an auditable SRe2L++ recovery manifest")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--recovery-seed", required=True, type=int)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--teacher-gate", required=True, type=Path)
    parser.add_argument("--patch-dir", required=True, type=Path)
    parser.add_argument("--patch-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--status", required=True, choices=("running", "complete"))
    args = parser.parse_args()
    cfg = get_dataset(args.dataset_name)
    for path in (args.teacher, args.teacher_gate, args.patch_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    patch_tree_hash, patch_files = tree_sha256(args.patch_dir)
    if patch_files != cfg.classes * 5:
        raise RuntimeError(f"Patch files {patch_files} != expected {cfg.classes * 5}")
    repo_root = Path(__file__).resolve().parents[2]
    recorded_revision = git_revision(repo_root)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    existing = {}
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
    started_revision = existing.get(
        "started_git_revision", existing.get("git_revision", recorded_revision)
    )
    fallback_started_at = (
        datetime.datetime.fromtimestamp(args.output.stat().st_mtime, datetime.timezone.utc).isoformat()
        if args.output.is_file()
        else now
    )
    started_at = existing.get("started_at", fallback_started_at)
    payload = {
        "status": args.status,
        "dataset": cfg.to_dict(),
        "recovery_seed": args.recovery_seed,
        "teacher": str(args.teacher.resolve()),
        "teacher_sha256": sha256(args.teacher),
        "teacher_gate": json.loads(args.teacher_gate.read_text(encoding="utf-8")),
        "patch_dir": str(args.patch_dir.resolve()),
        "patch_tree_sha256": patch_tree_hash,
        "patch_files": patch_files,
        "patch_manifest": json.loads(args.patch_manifest.read_text(encoding="utf-8")),
        "protocol": {
            "ipc_recovered": 5,
            "class_batch_size": 100,
            "optimizer": "Adam",
            "betas": [0.5, 0.9],
            "image_lr": 1e-3,
            "iterations": cfg.recovery_iterations,
            "r_bn": 1e-3,
            "first_bn_multiplier": 10.0,
            "jitter": 32,
            "augmentation": "RandomResizedCrop(224)+RandomHorizontalFlip",
            "initialization": "2x2 patches",
        },
        "git_revision": started_revision,
        "started_git_revision": started_revision,
        "recorded_git_revision": recorded_revision,
        "started_at": started_at,
    }
    if args.status == "complete":
        payload["completed_at"] = now
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "status": args.status,
        "dataset": cfg.name,
        "recovery_seed": args.recovery_seed,
        "teacher_sha256": payload["teacher_sha256"],
        "patch_tree_sha256": patch_tree_hash,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
