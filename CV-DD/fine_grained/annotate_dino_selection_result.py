"""Attach five-arm DINO selection provenance to a hard-label v1 result."""

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
    parser.add_argument("--selection-manifest", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("experiment") != "dino_fivearm_ipc1":
        raise RuntimeError("selection manifest is not a complete five-arm DINO manifest")
    audit_path = Path(manifest["selection_audit"])
    if not audit_path.is_file() or sha256(audit_path) != manifest["selection_audit_sha256"]:
        raise RuntimeError("selection geometry audit is missing or changed")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "complete":
        raise RuntimeError("selection geometry audit is incomplete")
    payload.update(
        method="DINORealSelection",
        supervision="hard_label_cross_entropy",
        selection_experiment="dino_fivearm_ipc1",
        selection_method=manifest["selection_method"],
        selection_arm=manifest["selection_arm"],
        selection_seed=manifest["selection_seed"],
        selection_manifest=str(args.selection_manifest.resolve()),
        selection_manifest_sha256=sha256(args.selection_manifest),
        selection_audit=str(audit_path.resolve()),
        selection_audit_sha256=sha256(audit_path),
        selected_tree_sha256=manifest["selected_tree_sha256"],
        dino_model_sha256=audit["dino_model_sha256"],
        dino_preprocessor_sha256=audit["dino_preprocessor_sha256"],
        eval_openblas_num_threads=int(os.environ.get("OPENBLAS_NUM_THREADS", "0")),
    )
    temporary = args.result.with_suffix(args.result.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.result)


if __name__ == "__main__":
    main()
