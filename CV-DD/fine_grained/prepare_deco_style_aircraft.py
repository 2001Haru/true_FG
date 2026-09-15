"""Controlled DeCO-style random-vs-TransFG region mosaics for Aircraft IPC3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from vit_teacher_common import IMAGENET_MEAN, IMAGENET_STD, atomic_json, build_teacher, sha256_file


def stable_digest(*values):
    return hashlib.sha256("\0".join(map(str, values)).encode("utf-8")).digest()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def boxes(path):
    result = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        result[fields[0]] = tuple(map(int, fields[1:5]))
    return result


@torch.no_grad()
def rollout(model, images):
    hidden = model.transformer.embeddings(images)
    n = hidden.shape[1]
    identity = torch.eye(n, device=hidden.device, dtype=torch.float32)[None]
    result = identity.expand(hidden.shape[0], -1, -1)
    for block in model.transformer.encoder.layer:
        hidden, weights = block(hidden)
        attention = weights.float().mean(1) + identity
        attention = attention / attention.sum(-1, keepdim=True)
        result = attention @ result
    scores = result[:, 0, 1:]
    if scores.shape[1] != 324:
        raise RuntimeError(f"unexpected patch grid: {scores.shape}")
    return F.interpolate(scores.reshape(-1, 1, 18, 18), (224, 224), mode="bilinear", align_corners=False)[:, 0]


def intersection(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def window_iou(a, b):
    overlap = intersection(a, b)
    return overlap / (2 * (a[2] - a[0]) * (a[3] - a[1]) - overlap)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--raw-images", required=True, type=Path)
    parser.add_argument("--boxes", required=True, type=Path)
    parser.add_argument("--transfg-source", required=True, type=Path)
    parser.add_argument("--transfg-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--selection-seed", default=20260914, type=int)
    parser.add_argument("--window", default=119, type=int)
    parser.add_argument("--batch-size", default=20, type=int)
    args = parser.parse_args()
    if args.window != 119:
        raise RuntimeError("first protocol requires a 119x119 reference window")
    selection = json.loads(args.selection_manifest.read_text())
    if (selection.get("status") != "complete" or selection.get("ipc") != 3
            or selection.get("selection_seed") not in (0, 1, 2)):
        raise RuntimeError("expected a completed RandomReal seed0/1/2 IPC3 manifest")
    box_map = boxes(args.boxes)
    r0_by_class = {}
    for row in selection["images"]:
        r0_by_class.setdefault(row["class_folder"], []).append(row)
    records = []
    for class_id, class_dir in enumerate(sorted(path for path in args.train_root.iterdir() if path.is_dir())):
        class_name = class_dir.name
        r0 = sorted(r0_by_class[class_name], key=lambda row: row["rank"])
        if len(r0) != 3:
            raise RuntimeError(f"class {class_name} lacks three R0 sources")
        r0_names = {Path(row["source_path"]).name for row in r0}
        candidates = sorted(
            (path for path in class_dir.iterdir() if path.is_file() and path.name not in r0_names),
            key=lambda path: stable_digest("deco-source", args.selection_seed, class_name, path.name),
        )
        extras = candidates[:9]
        if len(extras) != 9:
            raise RuntimeError(f"class {class_name} lacks nine extra sources")
        for mosaic in range(3):
            group = [Path(r0[mosaic]["source_path"])] + extras[mosaic * 3 : mosaic * 3 + 3]
            group = sorted(group, key=lambda path: stable_digest("deco-tile", args.selection_seed, class_name, mosaic, path.name))
            for tile, source in enumerate(group):
                image_id = source.stem
                raw = args.raw_images / f"{image_id}.jpg"
                if not raw.is_file() or image_id not in box_map:
                    raise RuntimeError(f"missing raw/bbox: {image_id}")
                records.append({
                    "class_id": class_id, "class": class_name, "mosaic": mosaic, "tile": tile,
                    "image_id": image_id, "raw_path": str(raw.resolve()),
                    "is_r0_source": source.name in r0_names,
                })
    if len(records) != 1200:
        raise RuntimeError(f"expected 1200 regions, got {len(records)}")
    model, architecture = build_teacher("transfg", args.transfg_source, weights_path=None, checkpoint_path=args.transfg_checkpoint)
    model.cuda().eval()
    side_positions = 224 - args.window + 1
    for offset in range(0, len(records), args.batch_size):
        batch = records[offset : offset + args.batch_size]
        images = []
        raw_sizes = []
        for row in batch:
            with Image.open(row["raw_path"]) as handle:
                raw = handle.convert("RGB")
            raw_sizes.append(raw.size)
            reference = raw.resize((224, 224), Image.Resampling.BILINEAR)
            images.append(TF.normalize(TF.pil_to_tensor(reference).float().div_(255), IMAGENET_MEAN, IMAGENET_STD))
        heatmaps = rollout(model, torch.stack(images).cuda()).cpu()
        pooled = F.avg_pool2d(heatmaps[:, None], args.window, stride=1)[:, 0]
        for row, heatmap, scores, raw_size in zip(batch, heatmaps, pooled, raw_sizes):
            fg_flat = int(scores.reshape(-1).argmax())
            fg_y, fg_x = divmod(fg_flat, side_positions)
            random_flat = int.from_bytes(stable_digest("deco-window", args.selection_seed, row["class"], row["image_id"])[:8], "little") % (side_positions * side_positions)
            random_y, random_x = divmod(random_flat, side_positions)
            fg = (fg_x, fg_y, fg_x + args.window, fg_y + args.window)
            random_window = (random_x, random_y, random_x + args.window, random_y + args.window)
            xmin, ymin, xmax, ymax = box_map[row["image_id"]]
            bbox = ((xmin - 1) / raw_size[0] * 224, (ymin - 1) / raw_size[1] * 224, xmax / raw_size[0] * 224, ymax / raw_size[1] * 224)
            row.update({
                "raw_size": list(raw_size), "official_bbox224": list(bbox),
                "fg_window224": list(fg), "random_window224": list(random_window),
                "fg_attention_mean": float(scores[fg_y, fg_x]),
                "random_attention_mean": float(scores[random_y, random_x]),
                "global_attention_mean": float(heatmap.mean()),
                "window_iou": window_iou(fg, random_window),
                "fg_bbox_intersection_over_window": intersection(fg, bbox) / (args.window ** 2),
                "random_bbox_intersection_over_window": intersection(random_window, bbox) / (args.window ** 2),
            })
    output_roots = {name: args.output_root / "selected" / name / "ipc3" for name in ("random_regions", "fg_regions")}
    for root in output_roots.values():
        root.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for row in records:
        grouped.setdefault((row["class"], row["mosaic"]), []).append(row)
    mosaic_records = []
    for (class_name, mosaic), group in sorted(grouped.items()):
        if len(group) != 4 or sorted(row["tile"] for row in group) != [0, 1, 2, 3]:
            raise RuntimeError("invalid mosaic group")
        outputs = {name: Image.new("RGB", (224, 224)) for name in output_roots}
        source_ids = []
        for row in sorted(group, key=lambda item: item["tile"]):
            source_ids.append(row["image_id"])
            with Image.open(row["raw_path"]) as handle:
                raw = handle.convert("RGB")
            for name, window_key in (("random_regions", "random_window224"), ("fg_regions", "fg_window224")):
                x0, y0, x1, y1 = row[window_key]
                raw_box = (x0 / 224 * raw.width, y0 / 224 * raw.height, x1 / 224 * raw.width, y1 / 224 * raw.height)
                tile = raw.transform((112, 112), Image.Transform.EXTENT, raw_box, Image.Resampling.BILINEAR)
                outputs[name].paste(tile, ((row["tile"] % 2) * 112, (row["tile"] // 2) * 112))
                row[name + "_raw_window"] = list(raw_box)
        output_paths = {}
        for name, image in outputs.items():
            destination = output_roots[name] / class_name / f"mosaic_{mosaic}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".png.tmp")
            image.save(temporary, format="PNG", compress_level=0)
            os.replace(temporary, destination)
            output_paths[name] = str(destination.resolve())
        mosaic_records.append({"class": class_name, "mosaic": mosaic, "source_ids": source_ids, "output_paths": output_paths})
    outputs = {}
    for name, root in output_roots.items():
        files = sorted(root.glob("*/*.png"))
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(bytes.fromhex(sha(path)))
        outputs[name] = {"root": str(root.resolve()), "images": len(files), "tree_sha256": digest.hexdigest()}
        if len(files) != 300:
            raise RuntimeError(f"{name} has {len(files)} mosaics")
    fg_scores = torch.tensor([row["fg_attention_mean"] for row in records], dtype=torch.float64)
    random_scores = torch.tensor([row["random_attention_mean"] for row in records], dtype=torch.float64)
    result = {
        "status": "complete", "protocol": "controlled_deco_style_regions_v1",
        "dataset": "A_imsize224", "ipc": 3, "classes": 100, "stored_images": 300,
        "regions": 1200, "regions_per_class": 12, "independent_sources_per_class": 12,
        "r0_sources_per_class": 3, "additional_sources_per_class": 9,
        "r0_selection_seed": int(selection["selection_seed"]),
        "selection_seed": args.selection_seed, "window_size_reference224": args.window,
        "window_area_ratio": args.window ** 2 / 224 ** 2, "tile_size": 112,
        "candidate_windows_per_source": side_positions ** 2,
        "candidate_rule": "all integer 119x119 windows fully inside the 224x224 full-image reference",
        "random_rule": "uniform stable-SHA256 index over the common candidate window list",
        "fg_rule": "maximum mean TransFG attention-rollout response over the same candidate list",
        "source_assignment": "each mosaic contains one distinct R0 source and three distinct additional sources; stable tile permutation",
        "pixel_rule": "selected reference window mapped to original raw pixels and sampled directly to 112x112 bilinear tile; 2x2 lossless PNG",
        "attention_rollout": "11 ordinary TransFG blocks; head mean; add identity; row normalize; layer product; CLS-to-patch; bilinear 18x18-to-224",
        "selection_manifest": str(args.selection_manifest.resolve()), "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "transfg_checkpoint": str(args.transfg_checkpoint.resolve()), "transfg_checkpoint_sha256": sha256_file(args.transfg_checkpoint),
        "architecture": architecture,
        "fg_minus_random_attention_mean": float((fg_scores - random_scores).mean()),
        "fg_attention_higher_fraction": float((fg_scores > random_scores).double().mean()),
        "fg_attention_mean": float(fg_scores.mean()), "random_attention_mean": float(random_scores.mean()),
        "window_iou_mean": sum(row["window_iou"] for row in records) / len(records),
        "fg_bbox_intersection_mean": sum(row["fg_bbox_intersection_over_window"] for row in records) / len(records),
        "random_bbox_intersection_mean": sum(row["random_bbox_intersection_over_window"] for row in records) / len(records),
        "outputs": outputs, "mosaics": mosaic_records, "regions_detail": records,
    }
    atomic_json(result, args.output_root / "construction_manifest.json")
    print(json.dumps({key: result[key] for key in ("status", "stored_images", "regions", "independent_sources_per_class", "window_area_ratio", "fg_attention_higher_fraction", "window_iou_mean", "fg_bbox_intersection_mean", "random_bbox_intersection_mean")}, indent=2))


if __name__ == "__main__":
    main()
