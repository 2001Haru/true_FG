"""Attach CoDA generation provenance to a standard hard-label result JSON."""

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
    parser.add_argument("--generation-audit", required=True, type=Path)
    parser.add_argument("--generation-config", required=True, type=Path)
    parser.add_argument("--generation-seed", required=True, type=int)
    args = parser.parse_args()
    for path in (args.result, args.generation_audit, args.generation_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    if payload.get("method") not in (None, "CoDA"):
        raise RuntimeError(f"Refusing to overwrite result method {payload.get('method')!r}")
    audit = json.loads(args.generation_audit.read_text(encoding="utf-8"))
    config = json.loads(args.generation_config.read_text(encoding="utf-8"))
    if audit.get("status") != "complete":
        raise RuntimeError("CoDA generation audit is incomplete")
    payload.update(
        method="CoDA",
        coda_feature_space=config["feature_space"],
        supervision="hard_label_cross_entropy",
        generation_seed=args.generation_seed,
        generation_audit=str(args.generation_audit.resolve()),
        generation_audit_sha256=sha256(args.generation_audit),
        generation_config=str(args.generation_config.resolve()),
        generation_config_sha256=sha256(args.generation_config),
        synthetic_tree_sha256=audit["tree_sha256"],
        cluster_audit_sha256=audit["cluster_audit_sha256"],
        generation_trace_sha256=audit["generation_trace_sha256"],
        generated_image_provenance_sha256=audit["generated_image_provenance_sha256"],
        coda_eval_openblas_num_threads=int(os.environ.get("OPENBLAS_NUM_THREADS", "0")),
    )
    temporary = args.result.with_suffix(args.result.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.result)


if __name__ == "__main__":
    main()
