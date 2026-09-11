"""Audit ImageNette same-source mosaic C/C native-CVDD results."""

import argparse
import hashlib
import json
import math
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


def expect_float(actual, expected: float, label: str, tolerance: float = 1e-12) -> None:
    if actual is None or not math.isclose(float(actual), expected, rel_tol=0, abs_tol=tolerance):
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--ipc", required=True, type=int, choices=(10, 50))
    parser.add_argument("--student-seed", required=True, type=int, choices=(42, 43, 44))
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--c-fkd", required=True, type=Path)
    parser.add_argument("--construction-manifest", required=True, type=Path)
    parser.add_argument("--metadata-audit", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    args = parser.parse_args()
    result = load(args.result)
    audit_payload(result, 10, 3925)
    expect(result["synthetic_data_path"], str(args.image_root.resolve()), "image root")
    expect(result["fkd_path"], str(args.c_fkd.resolve()), "C FKD")
    expect(result["student_protocol_name"], "imagenette_cvdd_native_resolution_v1", "protocol")
    expect(result["student_initialization"], "random", "initialization")
    expect(result["student_seed"], args.student_seed, "Student seed")
    expect(result["optimizer"], "adamw", "optimizer")
    expect_float(result["learning_rate"], 5e-4, "LR")
    expect_float(result["weight_decay"], 0.01, "WD")
    expect(result["epochs"], 300, "epochs")
    expect(result["batch_size"], 10, "batch")
    expect(result["gradient_accumulation_steps"], 2, "accumulation")
    expect(result["total_optimizer_updates"], 300 * args.ipc, "updates")
    expect_float(result["temperature"], 20, "temperature")
    expect(result["temperature_squared_multiplier"], False, "T squared")
    expect(result["mix_type"], "cutmix", "CutMix")
    expect(result["scheduler"], "cosine_lambda", "scheduler")
    eta = 1 if args.ipc == 10 else 2
    expect_float(result["cosine_eta"], eta, "eta")
    final_lr = 0 if args.ipc == 10 else 2.5e-4
    if any(not math.isclose(float(value), final_lr, rel_tol=0, abs_tol=1e-10)
           for value in result["final_scheduler_lrs"]):
        raise RuntimeError(f"final LR mismatch: {result['final_scheduler_lrs']}")
    construction = load(args.construction_manifest)
    expect(construction["status"], "complete", "construction")
    expect(construction["construction_seed"], 42, "construction seed")
    expect(construction["source_occurrences"], 4, "source occurrences")
    metadata = load(args.metadata_audit)
    expect(metadata["status"], "complete", "metadata replay")
    expect(metadata["total_mismatches"], 0, "metadata mismatches")
    relabel = load(args.c_fkd / "relabel_manifest.json")
    fkd = load(args.c_fkd / "fkd_audit.json")
    expect(relabel["status"], "complete", "C Relabel")
    expect(relabel["rescored_from_actual_replayed_views"], True, "rescoring")
    expect(relabel["teacher_mode"], "train", "BSSL")
    expect(relabel["teacher_forward_split"], [5, 5], "Teacher split")
    expect(relabel["teacher_sha256"], sha256(args.teacher), "Teacher hash")
    expect(fkd["status"], "complete", "FKD")
    expect(fkd["images"], 10 * args.ipc, "images")
    result["imagenette_same_source_mosaic"] = {
        "condition": "C image/C label", "ipc": args.ipc,
        "construction_seed": 42, "student_seed": args.student_seed,
        "source_occurrences_static": 4,
        "effective_exposure_claim": "not inferred after RRC/CutMix",
        "construction_manifest": str(args.construction_manifest.resolve()),
        "construction_manifest_sha256": sha256(args.construction_manifest),
        "metadata_replay_audit": str(args.metadata_audit.resolve()),
        "metadata_replay_audit_sha256": sha256(args.metadata_audit),
    }
    temporary = args.result.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.result)
    print(json.dumps({"status": "complete", "result": str(args.result.resolve())}))


if __name__ == "__main__":
    main()
