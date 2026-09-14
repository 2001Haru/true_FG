"""Build matched low-resolution/native-detail insets on R0 rank-2 images."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageStat
from torchvision.transforms import functional as TF

from vit_teacher_common import IMAGENET_MEAN, IMAGENET_STD, atomic_json, build_teacher, sha256_file


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", required=True, type=Path)
    parser.add_argument("--r0-root", required=True, type=Path)
    parser.add_argument("--raw-images", required=True, type=Path)
    parser.add_argument("--boxes", required=True, type=Path)
    parser.add_argument("--transfg-source", required=True, type=Path)
    parser.add_argument("--transfg-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--window", default=32, type=int)
    parser.add_argument("--inset", default=64, type=int)
    parser.add_argument("--batch-size", default=20, type=int)
    return parser.parse_args()


def file_sha(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_boxes(path: Path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        result[fields[0]] = tuple(map(int, fields[1:5]))
    return result


@torch.no_grad()
def attention_rollout(model, images):
    hidden = model.transformer.embeddings(images)
    tokens = hidden.shape[1]
    rollout = torch.eye(tokens, device=hidden.device, dtype=torch.float32)[None].expand(hidden.shape[0], -1, -1)
    identity = torch.eye(tokens, device=hidden.device, dtype=torch.float32)[None]
    for block in model.transformer.encoder.layer:
        hidden, weights = block(hidden)
        attention = weights.float().mean(dim=1) + identity
        attention = attention / attention.sum(dim=-1, keepdim=True)
        rollout = attention @ rollout
    patch_scores = rollout[:, 0, 1:]
    if patch_scores.shape[1] != 18 * 18:
        raise RuntimeError(f"expected 18x18 TransFG grid, got {patch_scores.shape}")
    return F.interpolate(
        patch_scores.reshape(-1, 1, 18, 18), size=(224, 224), mode="bilinear", align_corners=False
    )[:, 0]


def intersection_area(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def choose_window(heatmap, bbox, size):
    pooled = F.avg_pool2d(heatmap[None, None], kernel_size=size, stride=1)[0, 0]
    xmin = max(0, math.ceil(bbox[0]))
    ymin = max(0, math.ceil(bbox[1]))
    xmax = min(224 - size, math.floor(bbox[2] - size))
    ymax = min(224 - size, math.floor(bbox[3] - size))
    if xmin > xmax or ymin > ymax:
        raise RuntimeError(f"bbox cannot contain {size}x{size} window: {bbox}")
    region = pooled[ymin : ymax + 1, xmin : xmax + 1]
    flat = int(region.reshape(-1).argmax())
    width = region.shape[1]
    y = ymin + flat // width
    x = xmin + flat % width
    return (x, y, x + size, y + size), float(pooled[y, x]), float(heatmap.mean())


def choose_corner(bbox, window, inset):
    corners = ((0, 0), (224 - inset, 0), (0, 224 - inset), (224 - inset, 224 - inset))
    valid = []
    wx = (window[0] + window[2]) / 2
    wy = (window[1] + window[3]) / 2
    for priority, (x, y) in enumerate(corners):
        rectangle = (x, y, x + inset, y + inset)
        if intersection_area(rectangle, bbox) == 0:
            distance = ((x + inset / 2) - wx) ** 2 + ((y + inset / 2) - wy) ** 2
            valid.append((-distance, priority, rectangle))
    return min(valid)[2] if valid else None


def pixel_mae(left, right):
    return sum(ImageStat.Stat(ImageChops.difference(left, right)).mean) / 3


def main():
    args = parse_args()
    if args.window != 32 or args.inset != 64:
        raise RuntimeError("first protocol is frozen at window32/inset64")
    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    if selection.get("status") != "complete" or selection.get("ipc") != 3 or selection.get("selection_seed") != 0:
        raise RuntimeError("expected complete R0 seed0 IPC3 selection manifest")
    boxes = load_boxes(args.boxes)
    rank2 = sorted((row for row in selection["images"] if row["rank"] == 2), key=lambda row: row["class_id"])
    if len(rank2) != 100 or {row["class_id"] for row in rank2} != set(range(100)):
        raise RuntimeError("selection manifest lacks exactly one rank-2 image per class")
    model, architecture = build_teacher(
        "transfg", args.transfg_source, weights_path=None, checkpoint_path=args.transfg_checkpoint
    )
    model.cuda().eval()
    heatmaps = {}
    for offset in range(0, len(rank2), args.batch_size):
        batch = rank2[offset : offset + args.batch_size]
        tensors = []
        for row in batch:
            r0 = args.r0_root / row["class_folder"] / Path(row["source_path"]).name
            with Image.open(r0) as handle:
                image = handle.convert("RGB")
            if image.size != (224, 224):
                raise RuntimeError(f"R0 image is not 224: {r0}")
            tensors.append(TF.normalize(TF.pil_to_tensor(image).float().div_(255.0), IMAGENET_MEAN, IMAGENET_STD))
        maps = attention_rollout(model, torch.stack(tensors).cuda()).cpu()
        for row, heatmap in zip(batch, maps):
            heatmaps[(row["class_id"], Path(row["source_path"]).stem)] = heatmap
    output_roots = {
        "lowres_detail": args.output_root / "selected" / "lowres_detail" / "ipc3",
        "native_detail": args.output_root / "selected" / "native_detail" / "ipc3",
    }
    for root in output_roots.values():
        root.mkdir(parents=True, exist_ok=True)
    records = []
    modified_classes = set()
    patch_maes = []
    outside_max_differences = []
    for row in sorted(selection["images"], key=lambda item: (item["class_id"], item["rank"])):
        class_name = row["class_folder"]
        source_name = Path(row["source_path"]).name
        r0_path = args.r0_root / class_name / source_name
        if row["rank"] < 2:
            for root in output_roots.values():
                destination = root / class_name / source_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() or destination.is_symlink():
                    if not destination.is_symlink() or destination.resolve() != r0_path.resolve():
                        raise RuntimeError(f"collision: {destination}")
                else:
                    os.symlink(r0_path.resolve(), destination)
            records.append({"class_id": row["class_id"], "class": class_name, "rank": row["rank"], "image_id": Path(source_name).stem, "modified": False, "role": "unchanged_anchor"})
            continue
        image_id = Path(source_name).stem
        raw_path = args.raw_images / f"{image_id}.jpg"
        if image_id not in boxes or not raw_path.is_file():
            raise RuntimeError(f"missing raw image or bbox: {image_id}")
        with Image.open(r0_path) as handle:
            base = handle.convert("RGB")
        with Image.open(raw_path) as handle:
            raw = handle.convert("RGB")
        xmin, ymin, xmax, ymax = boxes[image_id]
        bbox224 = ((xmin - 1) / raw.width * 224, (ymin - 1) / raw.height * 224, xmax / raw.width * 224, ymax / raw.height * 224)
        window, selected_attention, global_attention = choose_window(
            heatmaps[(row["class_id"], image_id)], bbox224, args.window
        )
        inset_box = choose_corner(bbox224, window, args.inset)
        if inset_box is None:
            for root in output_roots.values():
                destination = root / class_name / source_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists() and not destination.is_symlink():
                    os.symlink(r0_path.resolve(), destination)
            records.append({"class_id": row["class_id"], "class": class_name, "rank": 2, "image_id": image_id, "modified": False, "reason": "no_bbox_disjoint_corner", "bbox224": list(bbox224), "window224": list(window)})
            continue
        low_patch = base.crop(window).resize((args.inset, args.inset), Image.Resampling.BILINEAR)
        raw_box = (
            window[0] / 224 * raw.width,
            window[1] / 224 * raw.height,
            window[2] / 224 * raw.width,
            window[3] / 224 * raw.height,
        )
        native_patch = raw.transform(
            (args.inset, args.inset), Image.Transform.EXTENT, raw_box, Image.Resampling.BILINEAR
        )
        outputs = {}
        for method, patch in (("lowres_detail", low_patch), ("native_detail", native_patch)):
            image = base.copy()
            image.paste(patch, (inset_box[0], inset_box[1]))
            destination = output_roots[method] / class_name / f"{image_id}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".png.tmp")
            image.save(temporary, format="PNG", compress_level=0)
            os.replace(temporary, destination)
            outputs[method] = image
            outside = ImageChops.difference(image, base)
            outside.paste((0, 0, 0), inset_box)
            outside_max_differences.append(max(channel[1] for channel in outside.getextrema()))
        if intersection_area(inset_box, bbox224) != 0:
            raise RuntimeError("selected inset overlaps official bbox")
        modified_classes.add(row["class_id"])
        patch_maes.append(pixel_mae(low_patch, native_patch))
        records.append({
            "class_id": row["class_id"], "class": class_name, "rank": 2, "image_id": image_id,
            "modified": True, "r0_path": str(r0_path.resolve()), "raw_path": str(raw_path.resolve()),
            "raw_size": list(raw.size), "official_bbox_1indexed": [xmin, ymin, xmax, ymax],
            "bbox224": list(bbox224), "window224": list(window), "raw_window": list(raw_box),
            "inset_box224": list(inset_box), "inset_bbox_overlap": 0,
            "selected_attention_mean": selected_attention, "global_attention_mean": global_attention,
            "attention_ratio": selected_attention / global_attention,
            "lowres_vs_native_patch_mae": patch_maes[-1],
        })
    if max(outside_max_differences, default=0) != 0:
        raise RuntimeError("pixels outside inset changed")
    manifests = {}
    for method, root in output_roots.items():
        files = sorted(path for path in root.glob("*/*") if path.is_file())
        if len(files) != 300 or len([path for path in root.iterdir() if path.is_dir()]) != 100:
            raise RuntimeError(f"invalid {method} output tree")
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(bytes.fromhex(file_sha(path.resolve())))
        manifests[method] = {"root": str(root.resolve()), "images": len(files), "tree_sha256": digest.hexdigest()}
    result = {
        "status": "complete", "protocol": "global_fullframe_plus_native_detail_v1",
        "selection_manifest": str(args.selection_manifest.resolve()), "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "r0_root": str(args.r0_root.resolve()), "raw_images": str(args.raw_images.resolve()),
        "boxes": str(args.boxes.resolve()), "transfg_checkpoint": str(args.transfg_checkpoint.resolve()),
        "transfg_checkpoint_sha256": sha256_file(args.transfg_checkpoint), "architecture": architecture,
        "attention_rollout": "11 ordinary blocks; mean heads; add identity; row normalize; A_l @ rollout; CLS-to-patch; bilinear 18x18-to-224",
        "window_rule": "highest mean-attention 32x32 integer window fully inside official bbox mapped to R0 224 coordinates",
        "inset_rule": "among bbox-disjoint 64x64 corners, farthest from selected-window center; tie priority TL,TR,BL,BR",
        "native_sampling": "float raw-image extent sampled directly to 64x64 with bilinear interpolation",
        "lowres_sampling": "same integer 32x32 window from decoded R0 224, bilinear resize to 64x64",
        "output_encoding": "modified rank-2 images saved lossless PNG; unchanged rank-0/1 and unavailable rank-2 symlink exact R0",
        "classes": 100, "images": 300, "modified_classes": len(modified_classes),
        "unchanged_classes": 100 - len(modified_classes), "modified_fraction_of_rank2": len(modified_classes) / 100,
        "inset_area_fraction": args.inset * args.inset / (224 * 224),
        "outside_inset_max_pixel_difference": max(outside_max_differences, default=0),
        "lowres_vs_native_patch_mae_mean": sum(patch_maes) / len(patch_maes) if patch_maes else None,
        "outputs": manifests, "records": records,
    }
    atomic_json(result, args.output_root / "construction_manifest.json")
    print(json.dumps({key: result[key] for key in ("status", "modified_classes", "unchanged_classes", "inset_area_fraction", "outside_inset_max_pixel_difference", "lowres_vs_native_patch_mae_mean")}, indent=2))
    if not modified_classes:
        raise RuntimeError("no class could be modified")


if __name__ == "__main__":
    main()
