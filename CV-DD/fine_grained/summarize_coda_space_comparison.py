"""Paired DINO-space minus VAE-space CoDA comparison over identical seeds."""

import argparse
import json
import os
import statistics
from pathlib import Path


DATASETS = ("CUB_imsize224", "A_imsize224", "SC_imsize224")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    errors = []
    comparison = {}
    paired = 0
    for dataset in DATASETS:
        comparison[dataset] = {}
        for ipc in (1, 3, 5):
            rows = []
            for generation_seed in (0, 1, 2):
                for student_seed in (42, 43, 44):
                    relative = Path("results") / dataset / f"ipc{ipc}_gseed{generation_seed}_sseed{student_seed}.json"
                    vae_path = args.base_root / "vae_space" / relative
                    dino_path = args.base_root / "dino_space" / relative
                    if not vae_path.is_file() or not dino_path.is_file():
                        errors.append(f"missing pair: {relative}")
                        continue
                    vae = json.loads(vae_path.read_text(encoding="utf-8"))
                    dino = json.loads(dino_path.read_text(encoding="utf-8"))
                    if vae.get("coda_feature_space") != "vae":
                        errors.append(f"invalid VAE provenance: {vae_path}")
                    if dino.get("coda_feature_space") != "dinov2":
                        errors.append(f"invalid DINO provenance: {dino_path}")
                    rows.append(
                        {
                            "generation_seed": generation_seed,
                            "student_seed": student_seed,
                            "vae_best_top1": float(vae["best_top1"]),
                            "dino_best_top1": float(dino["best_top1"]),
                            "delta_dino_minus_vae": float(dino["best_top1"] - vae["best_top1"]),
                        }
                    )
                    paired += 1
            if len(rows) == 9:
                deltas = [row["delta_dino_minus_vae"] for row in rows]
                comparison[dataset][str(ipc)] = {
                    "pairs": rows,
                    "delta_mean": statistics.mean(deltas),
                    "delta_sample_std": statistics.stdev(deltas),
                    "dino_wins": sum(delta > 0 for delta in deltas),
                    "ties": sum(delta == 0 for delta in deltas),
                    "vae_wins": sum(delta < 0 for delta in deltas),
                }
    payload = {
        "status": "complete" if paired == 81 and not errors else "incomplete",
        "comparison": "dino_space minus vae_space",
        "expected_pairs": 81,
        "paired_results": paired,
        "errors": errors,
        "results": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "complete":
        raise RuntimeError(f"CoDA feature-space comparison incomplete: {paired}/81 pairs")


if __name__ == "__main__":
    main()
