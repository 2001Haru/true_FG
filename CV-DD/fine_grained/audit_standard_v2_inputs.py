"""Preflight the 243 immutable v1 Teacher/Recovery/FKD inputs reused by v2."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from audit_result import audit_payload


DATASETS = {
    "CUB_imsize224": (200, 5794),
    "A_imsize224": (100, 3333),
    "SC_imsize224": (196, 8041),
}
SEEDS = (42, 43, 44)
IPCS = (1, 3, 5)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-root", required=True, type=Path)
    parser.add_argument("--cub4k-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows, errors = [], []
    checked_artifacts = {}
    for dataset, (classes, validation_images) in DATASETS.items():
        for teacher_seed in SEEDS:
            for recovery_seed in SEEDS:
                for ipc in IPCS:
                    for student_seed in SEEDS:
                        result = ((args.cub4k_root if dataset == "CUB_imsize224" else args.v1_root) /
                                  "results" / f"tseed{teacher_seed}" / dataset /
                                  f"rseed{recovery_seed}" / f"ipc{ipc}_sseed{student_seed}.json")
                        try:
                            payload = json.loads(result.read_text(encoding="utf-8"))
                            audit_payload(payload, classes, validation_images)
                            protocol = payload[
                                "cub4k_protocol" if dataset == "CUB_imsize224" else "standard_protocol"
                            ]
                            observed = tuple(protocol[key] for key in (
                                "teacher_seed", "recovery_seed", "ipc", "student_seed"
                            ))
                            expected = (teacher_seed, recovery_seed, ipc, student_seed)
                            if observed != expected:
                                raise RuntimeError(f"tuple mismatch {observed} != {expected}")
                            if dataset == "CUB_imsize224" and protocol["recovery_iterations"] != 4000:
                                raise RuntimeError("CUB reference is not the 4k matrix")
                            for path_key, hash_key in (
                                ("teacher_checkpoint", "teacher_checkpoint_sha256"),
                                ("recovery_manifest", "recovery_manifest_sha256"),
                                ("relabel_manifest", "relabel_manifest_sha256"),
                                ("fkd_audit", "fkd_audit_sha256"),
                            ):
                                path = Path(protocol[path_key])
                                expected_hash = protocol[hash_key]
                                identity = str(path.resolve())
                                if identity not in checked_artifacts:
                                    actual_hash = sha256(path)
                                    if actual_hash != expected_hash:
                                        raise RuntimeError(f"{path_key} SHA-256 mismatch: {path}")
                                    checked_artifacts[identity] = actual_hash
                            rows.append({
                                "dataset": dataset, "teacher_seed": teacher_seed,
                                "recovery_seed": recovery_seed, "ipc": ipc,
                                "student_seed": student_seed, "reference_result": str(result.resolve()),
                                "reference_result_sha256": sha256(result),
                            })
                        except Exception as error:
                            errors.append({"result": str(result.resolve()), "error": str(error)})
    payload = {
        "status": "complete" if len(rows) == 243 and not errors else "failed",
        "expected_reference_results": 243, "validated_reference_results": len(rows),
        "unique_hashed_upstream_artifacts": len(checked_artifacts),
        "recovery_iterations": {dataset: 4000 for dataset in DATASETS},
        "rows": rows, "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({
        "status": payload["status"], "validated": len(rows), "errors": len(errors),
        "output": str(args.output.resolve()),
    }))
    if payload["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
