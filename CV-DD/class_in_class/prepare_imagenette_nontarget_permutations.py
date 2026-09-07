"""Freeze one deterministic non-target class derangement per lambda0 image."""

import argparse
import json
from pathlib import Path

import torch

from imagenette_entropy_protocol import CLASSES, atomic_json, file_sha256, identity_seed


def fixed_derangement(relative_path, target):
    non_targets = [index for index in range(CLASSES) if index != target]
    for attempt in range(1000):
        generator = torch.Generator().manual_seed(
            identity_seed(
                "imagenette-nontarget-probability-derangement-v1",
                relative_path,
                attempt,
            )
        )
        sources = [non_targets[index] for index in torch.randperm(len(non_targets), generator=generator).tolist()]
        if all(destination != source for destination, source in zip(non_targets, sources)):
            mapping = list(range(CLASSES))
            for destination, source in zip(non_targets, sources):
                mapping[destination] = source
            return mapping, attempt
    raise RuntimeError(f"could not generate derangement for {relative_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    output = root / "preflight" / "lambda0_nontarget_permutations.json"
    if output.exists() and not args.force:
        raise RuntimeError(f"refusing overwrite: {output}")
    manifests = []
    rows_by_path = {}
    for selection_seed in (0, 1, 2):
        path = root / "manifests" / f"lambda_{0:+d}_rseed{selection_seed}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        manifests.append({"selection_seed": selection_seed, "path": str(path), "sha256": file_sha256(path)})
        for row in payload["images"]:
            existing = rows_by_path.get(row["relative_path"])
            if existing is not None and int(existing["class_id"]) != int(row["class_id"]):
                raise RuntimeError("same image has inconsistent labels across manifests")
            rows_by_path[row["relative_path"]] = row
    permutations = {}
    attempts = []
    for relative_path, row in sorted(rows_by_path.items()):
        target = int(row["class_id"])
        mapping, attempt = fixed_derangement(relative_path, target)
        attempts.append(attempt)
        permutations[relative_path] = {
            "class_id": target,
            "output_class_takes_input_class": mapping,
            "attempt": attempt,
        }
    unique = {tuple(row["output_class_takes_input_class"]) for row in permutations.values()}
    payload = {
        "status": "complete",
        "name": "imagenette_nontarget_probability_derangement_v1",
        "namespace": "imagenette-nontarget-probability-derangement-v1",
        "lambda0_manifests": manifests,
        "unique_images": len(permutations),
        "unique_full_class_mappings": len(unique),
        "generation": "torch.randperm rejection sampling with stable SHA-256 seed; frozen in this file",
        "true_class_fixed_for_every_image": True,
        "non_target_fixed_points": 0,
        "maximum_rejection_attempt": max(attempts),
        "permutations": permutations,
    }
    atomic_json(output, payload)
    print(json.dumps({key: payload[key] for key in ("status", "unique_images", "unique_full_class_mappings", "maximum_rejection_attempt")}, indent=2))


if __name__ == "__main__":
    main()

