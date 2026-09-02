"""Atomically attach FD2 provenance to an otherwise standard post-eval JSON."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--semantics", required=True, choices=("released_semantics", "paper_literal"))
    parser.add_argument("--joint-teacher", required=True, type=Path)
    parser.add_argument("--backbone", required=True, type=Path)
    parser.add_argument("--recovery-manifest", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.result, args.joint_teacher, args.backbone, args.recovery_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    existing_semantics = payload.get("fd2_semantics")
    if existing_semantics is not None and existing_semantics != args.semantics:
        raise RuntimeError(
            f"Refusing to overwrite result semantics {existing_semantics!r} with {args.semantics!r}"
        )
    payload["method"] = "FD2 on standard SRe2L++ foundation"
    payload["fd2_semantics"] = args.semantics
    payload["fd2_teacher"] = {
        "joint_path": str(args.joint_teacher.resolve()),
        "joint_sha256": sha256(args.joint_teacher),
        "backbone_path": str(args.backbone.resolve()),
        "backbone_sha256": sha256(args.backbone),
    }
    payload["fd2_recovery_manifest"] = str(args.recovery_manifest.resolve())
    temporary = args.result.with_suffix(args.result.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, args.result)


if __name__ == "__main__":
    main()
