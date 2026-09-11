"""Materialize nested DINO global-centroid Top1/3/5 ImageFolder selections."""

import argparse
import hashlib
import json
import os
from pathlib import Path


IPCS = (1, 3, 5)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top5-manifest", required=True, type=Path)
    parser.add_argument("--ipc1-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    top5 = json.loads(args.top5_manifest.read_text(encoding="utf-8"))
    ipc1 = json.loads(args.ipc1_manifest.read_text(encoding="utf-8"))
    if top5.get("status") != "complete" or top5.get("selection_method") != "global_center_top5":
        raise RuntimeError("invalid frozen global-center Top5 manifest")
    if top5.get("classes") != 100 or top5.get("selection_images") != 500:
        raise RuntimeError("unexpected Aircraft Top5 dimensions")
    if ipc1.get("status") != "complete" or ipc1.get("selection_method") != "centroid":
        raise RuntimeError("invalid frozen IPC1 centroid manifest")
    by_class = {class_id: [] for class_id in range(100)}
    for row in top5["images"]:
        by_class[int(row["class_id"])].append(row)
    old_ipc1 = {int(row["class_id"]): str(Path(row["source_path"]).resolve()) for row in ipc1["images"]}
    manifests = {}
    for ipc in IPCS:
        records = []
        selected_root = args.output_root / f"ipc{ipc}"
        for class_id in range(100):
            ranked = sorted(by_class[class_id], key=lambda row: int(row["slot"]))
            if len(ranked) != 5 or [int(row["slot"]) for row in ranked] != list(range(5)):
                raise RuntimeError(f"class {class_id}: invalid Top5 ranks")
            if str(Path(ranked[0]["source_path"]).resolve()) != old_ipc1[class_id]:
                raise RuntimeError(f"class {class_id}: rank1 differs from frozen IPC1 centroid")
            class_folder = ranked[0]["class_folder"]
            destination_class = selected_root / class_folder
            destination_class.mkdir(parents=True, exist_ok=True)
            for rank, row in enumerate(ranked[:ipc]):
                source = Path(row["source_path"]).resolve()
                destination = destination_class / f"rank{rank:02d}_{source.name}"
                if destination.exists() or destination.is_symlink():
                    if not destination.is_symlink() or destination.resolve() != source:
                        raise RuntimeError(f"destination collision: {destination}")
                else:
                    os.symlink(source, destination)
                records.append({
                    "class_id": class_id, "class_folder": class_folder, "rank": rank + 1,
                    "source_path": str(source), "source_sha256": sha256(source),
                    "selected_path": str(destination.absolute()),
                    "own_centroid_similarity": float(row["own_centroid_similarity"]),
                    "selection_role": row["selection_role"],
                })
        payload = {
            "status": "complete", "dataset": "A_imsize224", "classes": 100, "ipc": ipc,
            "method": "dino_global_centroid_top_ipc", "selection_seed": None,
            "geometry": "L2-normalized DINOv2 CLS; normalized class centroid; cosine similarity descending",
            "selected_images": len(records), "selected_root": str(selected_root.resolve()),
            "source_top5_manifest": str(args.top5_manifest.resolve()),
            "source_top5_manifest_sha256": sha256(args.top5_manifest),
            "frozen_ipc1_manifest": str(args.ipc1_manifest.resolve()),
            "frozen_ipc1_manifest_sha256": sha256(args.ipc1_manifest),
            "ipc1_rank_identity_verified": True, "nested_top_ipc": True, "images": records,
        }
        manifest_path = args.output_root / "manifests" / f"ipc{ipc}.json"
        atomic_json(payload, manifest_path)
        manifests[str(ipc)] = {"manifest": str(manifest_path.resolve()), "selected_images": len(records)}
    sets = {}
    for ipc in IPCS:
        manifest = json.loads((args.output_root / "manifests" / f"ipc{ipc}.json").read_text())
        sets[ipc] = {(row["class_id"], row["source_sha256"]) for row in manifest["images"]}
    if not sets[1] <= sets[3] <= sets[5]:
        raise RuntimeError("nested IPC identity audit failed")
    atomic_json({"status": "complete", "ipcs": list(IPCS), "counts": {str(k): len(v) for k, v in sets.items()},
                 "ipc1_subset_ipc3_subset_ipc5": True, "manifests": manifests},
                args.output_root / "selection_audit.json")


if __name__ == "__main__":
    main()
