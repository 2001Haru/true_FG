"""Attach IPC2 selection provenance to a hard-label v1 result."""

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
    parser.add_argument("--selection-audit", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    audit = json.loads(args.selection_audit.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("experiment") != "dino_ipc2_center_complement"
        or audit.get("status") != "complete"
        or audit.get("experiment") != "dino_ipc2_center_complement"
    ):
        raise RuntimeError("IPC2 selection provenance is incomplete")
    if audit.get("manifests", {}).get(manifest["selection_arm"]) != str(
        args.selection_manifest.resolve()
    ):
        raise RuntimeError("selection manifest is not registered by the IPC2 audit")
    payload.update(
        method="DINOIPC2Selection",
        supervision="hard_label_cross_entropy",
        selection_experiment=manifest["experiment"],
        selection_method=manifest["selection_method"],
        selection_arm=manifest["selection_arm"],
        selection_seed=manifest["selection_seed"],
        selection_manifest=str(args.selection_manifest.resolve()),
        selection_manifest_sha256=sha256(args.selection_manifest),
        selection_audit=str(args.selection_audit.resolve()),
        selection_audit_sha256=sha256(args.selection_audit),
        selected_path_identity_sha256=manifest["selected_path_identity_sha256"],
        parent_ipc1_geometry=manifest["parent_ipc1_geometry"],
        geometry_recomputed=False,
        training_sample_weighting=manifest["training_sample_weighting"],
        eval_openblas_num_threads=int(os.environ.get("OPENBLAS_NUM_THREADS", "0")),
    )
    temporary = args.result.with_suffix(args.result.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.result)


if __name__ == "__main__":
    main()
