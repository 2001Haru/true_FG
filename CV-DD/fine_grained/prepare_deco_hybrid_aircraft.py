"""Build one-full-plus-two-mosaic Aircraft IPC3 trees from frozen DeCO-style artifacts."""
import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(root):
    h = hashlib.sha256()
    for path in sorted(p for p in Path(root).rglob("*") if p.is_file()):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(bytes.fromhex(sha256(path)))
    return h.hexdigest()


def link(source, destination):
    source, destination = Path(source).resolve(), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        if destination.resolve() != source:
            raise RuntimeError(f"collision at {destination}: {destination.resolve()} != {source}")
        return
    destination.symlink_to(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--construction-manifest", required=True, type=Path)
    parser.add_argument("--r0-selection-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    source = json.loads(args.construction_manifest.read_text())
    selection = json.loads(args.r0_selection_manifest.read_text())
    assert source["status"] == selection["status"] == "complete"
    by_class = defaultdict(list)
    for row in source["regions_detail"]:
        by_class[row["class"]].append(row)
    selected = {(row["class_folder"], Path(row["selected_path"]).stem): Path(row["selected_path"])
                for row in selection["images"]}
    output_roots = {name: args.output_root / f"selected/{name}/ipc3"
                    for name in ("hybrid_fg", "hybrid_random")}
    classes = []
    for class_name in sorted(by_class):
        rows = by_class[class_name]
        omitted = [r for r in rows if r["mosaic"] == 2]
        kept = [r for r in rows if r["mosaic"] in (0, 1)]
        full_rows = [r for r in omitted if r["is_r0_source"]]
        assert len(rows) == 12 and len(omitted) == 4 and len(kept) == 8 and len(full_rows) == 1
        full = full_rows[0]
        ids = [full["image_id"]] + [r["image_id"] for r in kept]
        assert len(ids) == len(set(ids)) == 9
        full_source = selected[(class_name, full["image_id"])]
        assert full_source.is_file()
        for method, source_method in (("hybrid_fg", "fg_regions"), ("hybrid_random", "random_regions")):
            root = output_roots[method] / class_name
            link(full_source, root / "slot0_full.jpg")
            for slot, mosaic in enumerate((0, 1), start=1):
                mosaic_source = Path(source["outputs"][source_method]["root"]) / class_name / f"mosaic_{mosaic}.png"
                assert mosaic_source.is_file()
                link(mosaic_source, root / f"slot{slot}_mosaic.png")
        classes.append({"class": class_name, "full_source_id": full["image_id"],
                        "full_source_path": str(full_source), "omitted_mosaic": 2,
                        "kept_mosaics": [0, 1], "tile_source_ids": [r["image_id"] for r in kept],
                        "independent_source_ids": ids})
    assert len(classes) == 100
    outputs = {name: {"root": str(root.resolve()), "images": sum(1 for p in root.rglob("*") if p.is_file()),
                      "tree_sha256": tree_sha256(root)} for name, root in output_roots.items()}
    assert all(item["images"] == 300 for item in outputs.values())
    manifest = {"status": "complete", "protocol": "deco_hybrid_one_full_two_mosaics_v1",
                "construction_manifest": str(args.construction_manifest.resolve()),
                "construction_manifest_sha256": sha256(args.construction_manifest),
                "r0_selection_manifest": str(args.r0_selection_manifest.resolve()),
                "r0_selection_manifest_sha256": sha256(args.r0_selection_manifest),
                "classes": 100, "ipc": 3, "stored_images": 300,
                "slots_per_class": {"full": 1, "mosaic": 2}, "independent_sources_per_class": 9,
                "slot_rule": "keep mosaics 0/1; use the unique R0 source assigned to omitted mosaic 2 as slot0_full",
                "outputs": outputs, "classes_detail": classes}
    out = args.output_root / "construction_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp"); tmp.write_text(json.dumps(manifest, indent=2) + "\n"); os.replace(tmp, out)
    print(json.dumps({k: manifest[k] for k in ("status", "classes", "stored_images", "independent_sources_per_class", "outputs")}, indent=2))


if __name__ == "__main__":
    main()
