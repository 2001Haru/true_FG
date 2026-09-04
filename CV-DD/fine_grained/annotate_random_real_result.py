"""Attach deterministic random-real selection provenance to a hard-label result."""

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
    parser.add_argument("--selection-seed", required=True, type=int)
    args = parser.parse_args()
    for path in (args.result, args.selection_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    manifest = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise RuntimeError("Random-real selection manifest is incomplete")
    if int(manifest["selection_seed"]) != args.selection_seed:
        raise RuntimeError("Selection seed differs from manifest")
    payload.update(
        method="RandomReal",
        supervision="hard_label_cross_entropy",
        selection_seed=args.selection_seed,
        selection_algorithm=manifest["selection_algorithm"],
        selection_manifest=str(args.selection_manifest.resolve()),
        selection_manifest_sha256=sha256(args.selection_manifest),
        selected_tree_sha256=manifest["selected_tree_sha256"],
        eval_openblas_num_threads=int(os.environ.get("OPENBLAS_NUM_THREADS", "0")),
    )
    temporary = args.result.with_suffix(args.result.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.result)


if __name__ == "__main__":
    main()
