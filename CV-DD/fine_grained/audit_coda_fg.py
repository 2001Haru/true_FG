"""Audit a fine-grained CoDA ImageFolder and write an atomic manifest."""

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from PIL import Image

from config import get_dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--synthetic-dir", required=True, type=Path)
    parser.add_argument("--ipc", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--generation-config", required=True, type=Path)
    parser.add_argument("--cluster-audit", required=True, type=Path)
    parser.add_argument("--generation-trace", required=True, type=Path)
    parser.add_argument("--generated-provenance-output", required=True, type=Path)
    args = parser.parse_args()
    cfg = get_dataset(args.dataset_name)
    expected_classes = sorted(path.name for path in (args.data_dir / "train").iterdir() if path.is_dir())
    actual_classes = sorted(path.name for path in args.synthetic_dir.iterdir() if path.is_dir())
    if len(expected_classes) != cfg.classes or actual_classes != expected_classes:
        raise RuntimeError("CoDA class folders do not exactly match the prepared training ImageFolder")
    if not args.generation_config.is_file():
        raise FileNotFoundError(args.generation_config)
    for path in (args.cluster_audit, args.generation_trace):
        if not path.is_file():
            raise FileNotFoundError(path)
    generation_config = json.loads(args.generation_config.read_text(encoding="utf-8"))
    cluster_audit = json.loads(args.cluster_audit.read_text(encoding="utf-8"))
    generation_trace = json.loads(args.generation_trace.read_text(encoding="utf-8"))
    if cluster_audit.get("status") != "complete":
        raise RuntimeError("Cluster audit is incomplete")
    if generation_trace.get("status") != "complete":
        raise RuntimeError("Generation trace is incomplete")
    if generation_trace.get("feature_space") != generation_config["feature_space"]:
        raise RuntimeError("Generation trace feature space differs from frozen config")
    if int(generation_trace.get("generation_seed")) != int(generation_config["generation_seed"]):
        raise RuntimeError("Generation trace seed differs from frozen config")

    representatives = {}
    provenance_files = cluster_audit.get("per_image_provenance_sha256", {})
    for provenance_name, expected_hash in provenance_files.items():
        provenance_path = Path(provenance_name)
        if sha256(provenance_path) != expected_hash:
            raise RuntimeError(f"Cluster provenance hash changed: {provenance_path}")
        with provenance_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["selected_as_representative"]:
                    key = (row["class_folder"], int(row["representative_slot"]))
                    if key in representatives:
                        raise RuntimeError(f"Duplicate representative provenance: {key}")
                    representatives[key] = row

    trace_by_key = {}
    for row in generation_trace.get("images", []):
        key = (row["class_folder"], int(row["representative_slot"]))
        if key in trace_by_key:
            raise RuntimeError(f"Duplicate generation trace entry: {key}")
        expected_seed = (
            int(generation_config["generation_seed"])
            + int(row["gpu_id"]) * 10000
            + int(row["class_id"]) * args.ipc
            + 1000
            + int(row["representative_slot"])
        )
        if int(row["image_seed"]) != expected_seed:
            raise RuntimeError(f"Generation seed formula mismatch: {key}")
        trace_by_key[key] = row
    if set(trace_by_key) != set(representatives):
        raise RuntimeError("Generation trace and selected representatives differ")

    tree_digest = hashlib.sha256()
    files = 0
    generated_rows = []
    for class_name in actual_classes:
        class_dir = args.synthetic_dir / class_name
        images = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(images) != args.ipc:
            raise RuntimeError(f"{class_name} contains {len(images)} images, expected IPC={args.ipc}")
        for path in images:
            relative = path.relative_to(args.synthetic_dir).as_posix()
            digest = sha256(path)
            tree_digest.update(relative.encode("utf-8"))
            tree_digest.update(bytes.fromhex(digest))
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if image.mode != "RGB" or image.size != (224, 224):
                    raise RuntimeError(f"Invalid CoDA image {path}: mode={image.mode}, size={image.size}")
            slot = int(path.stem)
            key = (class_name, slot)
            if key not in representatives or key not in trace_by_key:
                raise RuntimeError(f"Generated image has no representative/trace: {relative}")
            representative = representatives[key]
            trace = trace_by_key[key]
            if Path(trace["output_path"]).resolve() != path.resolve():
                raise RuntimeError(f"Generation trace output path mismatch: {relative}")
            generated_rows.append({
                "schema_version": 1,
                "generated_relative_path": relative,
                "generated_path": str(path.resolve()),
                "generated_sha256": digest,
                "generation_seed": int(generation_config["generation_seed"]),
                "image_seed": int(trace["image_seed"]),
                "gpu_id": int(trace["gpu_id"]),
                "class_id": int(trace["class_id"]),
                "class_folder": class_name,
                "prompt": trace["prompt"],
                "representative_slot": slot,
                "representative_source_path": representative["source_path"],
                "representative_initial_hdbscan_label": representative["initial_hdbscan_label"],
                "representative_initial_hdbscan_is_noise": representative["initial_hdbscan_is_noise"],
                "representative_initial_hdbscan_membership_probability": representative[
                    "initial_hdbscan_membership_probability"
                ],
                "representative_initial_hdbscan_outlier_score": representative[
                    "initial_hdbscan_outlier_score"
                ],
                "representative_origin": representative["representative_origin"],
                "representative_selection": representative["representative_selection"],
            })
            files += 1
    if files != cfg.classes * args.ipc:
        raise RuntimeError(f"Generated image total {files} != {cfg.classes * args.ipc}")
    args.generated_provenance_output.parent.mkdir(parents=True, exist_ok=True)
    generated_tmp = args.generated_provenance_output.with_suffix(
        args.generated_provenance_output.suffix + ".tmp"
    )
    with generated_tmp.open("w", encoding="utf-8") as handle:
        for row in sorted(
            generated_rows, key=lambda item: (item["class_id"], item["representative_slot"])
        ):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(generated_tmp, args.generated_provenance_output)
    payload = {
        "status": "complete",
        "method": "CoDA fine-grained adapter",
        "feature_space": generation_config["feature_space"],
        "dataset": cfg.name,
        "classes": cfg.classes,
        "ipc": args.ipc,
        "files": files,
        "image_mode": "RGB",
        "image_size": [224, 224],
        "synthetic_dir": str(args.synthetic_dir.resolve()),
        "tree_sha256": tree_digest.hexdigest(),
        "generation_config": str(args.generation_config.resolve()),
        "generation_config_sha256": sha256(args.generation_config),
        "cluster_audit": str(args.cluster_audit.resolve()),
        "cluster_audit_sha256": sha256(args.cluster_audit),
        "generation_trace": str(args.generation_trace.resolve()),
        "generation_trace_sha256": sha256(args.generation_trace),
        "generated_image_provenance": str(args.generated_provenance_output.resolve()),
        "generated_image_provenance_sha256": sha256(args.generated_provenance_output),
        "generated_images_from_initial_noise": sum(
            row["representative_initial_hdbscan_is_noise"] for row in generated_rows
        ),
        "generated_images_by_representative_origin": dict(sorted(
            Counter(row["representative_origin"] for row in generated_rows).items()
        )),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
