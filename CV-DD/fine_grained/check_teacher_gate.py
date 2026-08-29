import argparse
import hashlib
import json
import os
from pathlib import Path

from config import get_dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser("Enforce the FD2 teacher accuracy gate before recovery")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--teacher-dir", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=3.0,
                        help="allowed Top-1 shortfall from the FD2 CAL-backbone reference")
    args = parser.parse_args()
    cfg = get_dataset(args.dataset_name)
    completion_path = args.teacher_dir / "complete.json"
    checkpoint_path = args.teacher_dir / "ResNet18.pth"
    if not completion_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("Teacher completion/checkpoint is missing")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    accuracy = float(completion["best_validation_accuracy"])
    threshold = cfg.teacher_reference_top1 - args.tolerance
    passed = accuracy >= threshold
    payload = {
        "dataset": cfg.name,
        "teacher_dir": str(args.teacher_dir.resolve()),
        "teacher_sha256": sha256(checkpoint_path),
        "best_validation_accuracy": accuracy,
        "best_epoch": completion["best_epoch"],
        "reference_top1": cfg.teacher_reference_top1,
        "tolerance": args.tolerance,
        "threshold": threshold,
        "passed": passed,
    }
    output = args.teacher_dir / "teacher_gate.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(payload, sort_keys=True))
    if not passed:
        raise RuntimeError(
            f"Teacher gate failed: {accuracy:.3f} < {threshold:.3f} "
            f"(FD2 reference {cfg.teacher_reference_top1:.3f}, tolerance {args.tolerance:.3f})"
        )


if __name__ == "__main__":
    main()
