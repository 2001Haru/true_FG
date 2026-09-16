"""Build five matched Aircraft IPC3 encode/decode compression diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from prepare_deco_style_aircraft import rollout
from vit_teacher_common import IMAGENET_MEAN, IMAGENET_STD, atomic_json, build_teacher, sha256_file


METHODS = ("isotropic158", "vertical112", "horizontal112", "bbox_retarget112", "attention_retarget112")


def stats(values):
    values = list(map(float, values))
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "min": min(values), "max": max(values)}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_png(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG", compress_level=0)
    os.replace(temporary, path)


def allocate_density(weights, target=112.0, minimum=0.1, maximum=0.9):
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise RuntimeError("invalid density weights")
    weights /= weights.sum()
    lo, hi = 0.0, target / max(weights.min(), 1e-12)
    for _ in range(100):
        mid = (lo + hi) / 2
        density = np.clip(minimum + mid * weights, minimum, maximum)
        if density.sum() < target:
            lo = mid
        else:
            hi = mid
    density = np.clip(minimum + hi * weights, minimum, maximum)
    density *= target / density.sum()
    if abs(density.sum() - target) > 1e-8 or density.min() < minimum - 1e-7 or density.max() > maximum + 1e-7:
        raise RuntimeError((density.sum(), density.min(), density.max()))
    return density


def bbox_density(y0, y1):
    first = max(0, min(223, math.floor(y0)))
    last = max(first + 1, min(224, math.ceil(y1)))
    subject = np.zeros(224, dtype=bool)
    subject[first:last] = True
    height = int(subject.sum())
    background = 224 - height
    if background == 0:
        return np.full(224, 0.5), height, True
    subject_density = min(0.9, (112.0 - 0.1 * background) / height)
    background_density = (112.0 - subject_density * height) / background
    density = np.where(subject, subject_density, background_density).astype(np.float64)
    if density.min() < 0.1 - 1e-8 or density.max() > 0.9 + 1e-8 or abs(density.sum() - 112) > 1e-8:
        raise RuntimeError((height, density.min(), density.max(), density.sum()))
    return density, height, subject_density < 0.9 - 1e-10


def encode_vertical_area(array, density):
    """Area-average source rows over inverse nonuniform-map intervals."""
    source = array.astype(np.float64)
    mapping = np.concatenate(([0.0], np.cumsum(density)))
    source_edges = np.arange(225, dtype=np.float64)
    inverse_edges = np.interp(np.arange(113, dtype=np.float64), mapping, source_edges)
    encoded = np.empty((112, source.shape[1], source.shape[2]), dtype=np.float64)
    for out_y, (lo, hi) in enumerate(zip(inverse_edges[:-1], inverse_edges[1:])):
        first, last = int(math.floor(lo)), int(math.ceil(hi))
        indices = np.arange(max(0, first), min(224, last), dtype=int)
        overlap = np.maximum(0.0, np.minimum(indices + 1.0, hi) - np.maximum(indices, lo))
        encoded[out_y] = np.tensordot(overlap / overlap.sum(), source[indices], axes=(0, 0))
    return np.rint(encoded).clip(0, 255).astype(np.uint8), mapping


def decode_vertical(encoded, mapping):
    """Restore each original row at its nonuniform encoded coordinate."""
    source = encoded.astype(np.float64)
    centers = np.interp(np.arange(224, dtype=np.float64) + 0.5, np.arange(225), mapping) - 0.5
    centers = np.clip(centers, 0.0, 111.0)
    low = np.floor(centers).astype(int)
    high = np.minimum(low + 1, 111)
    alpha = (centers - low)[:, None, None]
    decoded = source[low] * (1.0 - alpha) + source[high] * alpha
    return np.rint(decoded).clip(0, 255).astype(np.uint8)


def attention_densities(records, source_root, transfg_source, checkpoint, batch_size):
    model, architecture = build_teacher("transfg", transfg_source, weights_path=None, checkpoint_path=checkpoint)
    model.cuda().eval()
    result = {}
    kernel_x = torch.arange(-20, 21, dtype=torch.float64)
    kernel = torch.exp(-0.5 * (kernel_x / 5.0).square())
    kernel /= kernel.sum()
    for offset in range(0, len(records), batch_size):
        batch = records[offset:offset + batch_size]
        tensors = []
        for row in batch:
            with Image.open(source_root / row["class"] / row["filename"]) as handle:
                image = handle.convert("RGB")
            tensor = TF.pil_to_tensor(image).float().div_(255.0)
            tensors.append(TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD))
        with torch.no_grad():
            heat = rollout(model, torch.stack(tensors).cuda()).double().cpu()
        row_scores = heat.mean(2)
        padded = F.pad(row_scores[:, None], (20, 20), mode="reflect")
        smooth = F.conv1d(padded, kernel[None, None])[:, 0]
        for row, score in zip(batch, smooth.numpy()):
            result[(row["class"], row["filename"])] = allocate_density(score)
    del model
    torch.cuda.empty_cache()
    return result, architecture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0-root", required=True, type=Path)
    parser.add_argument("--raw-images", required=True, type=Path)
    parser.add_argument("--boxes", required=True, type=Path)
    parser.add_argument("--transfg-source", required=True, type=Path)
    parser.add_argument("--transfg-checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    boxes = {}
    for line in args.boxes.read_text().splitlines():
        fields = line.split()
        boxes[fields[0]] = tuple(map(int, fields[1:5]))
    records = []
    for class_dir in sorted(path for path in args.r0_root.iterdir() if path.is_dir()):
        for source in sorted(class_dir.iterdir()):
            image_id = source.stem
            raw_path = args.raw_images / f"{image_id}.jpg"
            if image_id not in boxes or not raw_path.is_file():
                raise RuntimeError(f"missing raw/bbox {image_id}")
            with Image.open(raw_path) as handle:
                raw_size = handle.size
            xmin, ymin, xmax, ymax = boxes[image_id]
            y0 = (ymin - 1.0) / raw_size[1] * 224.0
            y1 = ymax / raw_size[1] * 224.0
            x0 = (xmin - 1.0) / raw_size[0] * 224.0
            x1 = xmax / raw_size[0] * 224.0
            records.append({"class": class_dir.name, "filename": source.name, "image_id": image_id,
                            "source_path": str(source.resolve()), "raw_size": list(raw_size),
                            "official_bbox224": [x0, y0, x1, y1]})
    if len(records) != 300:
        raise RuntimeError(f"expected 300 R0 images, got {len(records)}")
    attention, architecture = attention_densities(
        records, args.r0_root, args.transfg_source, args.transfg_checkpoint, args.batch_size
    )
    method_metrics = {method: {"mae": [], "rmse": []} for method in METHODS}
    bbox_heights, bbox_subject_density, bbox_background_density, bbox_reduced = [], [], [], 0
    attention_min, attention_max, attention_entropy = [], [], []
    output_records = []
    for row in records:
        source_path = args.r0_root / row["class"] / row["filename"]
        with Image.open(source_path) as handle:
            original = handle.convert("RGB")
        if original.size != (224, 224):
            raise RuntimeError(f"R0 not 224: {source_path}")
        array = np.asarray(original, dtype=np.uint8)
        bbox_d, bbox_h, reduced = bbox_density(row["official_bbox224"][1], row["official_bbox224"][3])
        attn_d = attention[(row["class"], row["filename"])]
        bbox_heights.append(bbox_h); bbox_reduced += int(reduced)
        bbox_subject_density.append(float(bbox_d.max()))
        bbox_background_density.append(float(bbox_d.min()))
        attention_min.append(float(attn_d.min())); attention_max.append(float(attn_d.max()))
        normalized = attn_d / attn_d.sum()
        attention_entropy.append(float(-(normalized * np.log(normalized)).sum()))
        generated = {}
        simple = {
            "isotropic158": ((158, 158), (224, 224)),
            "vertical112": ((224, 112), (224, 224)),
            "horizontal112": ((112, 224), (224, 224)),
        }
        for method, (encoded_size, decoded_size) in simple.items():
            encoded = original.resize(encoded_size, Image.Resampling.LANCZOS)
            encoded_path = args.output_root / "encoded" / method / row["class"] / (Path(row["filename"]).stem + ".png")
            save_png(encoded, encoded_path)
            with Image.open(encoded_path) as handle:
                reloaded = handle.convert("RGB")
            decoded = reloaded.resize(decoded_size, Image.Resampling.LANCZOS)
            generated[method] = (encoded_path, decoded)
        for method, density in (("bbox_retarget112", bbox_d), ("attention_retarget112", attn_d)):
            encoded_array, mapping = encode_vertical_area(array, density)
            encoded_path = args.output_root / "encoded" / method / row["class"] / (Path(row["filename"]).stem + ".png")
            save_png(Image.fromarray(encoded_array, "RGB"), encoded_path)
            with Image.open(encoded_path) as handle:
                reloaded = np.asarray(handle.convert("RGB"), dtype=np.uint8)
            decoded = Image.fromarray(decode_vertical(reloaded, mapping), "RGB")
            generated[method] = (encoded_path, decoded)
        paths = {}
        for method, (encoded_path, decoded) in generated.items():
            decoded_path = args.output_root / "selected" / method / "ipc3" / row["class"] / (Path(row["filename"]).stem + ".png")
            save_png(decoded, decoded_path)
            decoded_array = np.asarray(decoded, dtype=np.float64)
            delta = decoded_array - array.astype(np.float64)
            method_metrics[method]["mae"].append(float(np.abs(delta).mean()))
            method_metrics[method]["rmse"].append(float(np.square(delta).mean() ** 0.5))
            paths[method] = {"encoded": str(encoded_path.resolve()), "decoded": str(decoded_path.resolve()),
                             "encoded_sha256": sha256(encoded_path), "decoded_sha256": sha256(decoded_path)}
        output_records.append({**row, "bbox_rows": bbox_h, "bbox_subject_density": float(bbox_d.max()),
                               "bbox_background_density": float(bbox_d.min()), "bbox_density_reduced": reduced,
                               "attention_density_min": float(attn_d.min()), "attention_density_max": float(attn_d.max()),
                               "outputs": paths})
    outputs = {}
    for method in METHODS:
        files = sorted((args.output_root / "selected" / method / "ipc3").glob("*/*.png"))
        encoded_files = sorted((args.output_root / "encoded" / method).glob("*/*.png"))
        if len(files) != 300 or len(encoded_files) != 300:
            raise RuntimeError(f"{method}: decoded={len(files)} encoded={len(encoded_files)}")
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.relative_to(args.output_root).as_posix().encode())
            digest.update(bytes.fromhex(sha256(path)))
        outputs[method] = {"decoded_root": str((args.output_root / "selected" / method / "ipc3").resolve()),
                           "encoded_root": str((args.output_root / "encoded" / method).resolve()),
                           "images": 300, "tree_sha256": digest.hexdigest(),
                           "pixel_mae_8bit": stats(method_metrics[method]["mae"]),
                           "pixel_rmse_8bit": stats(method_metrics[method]["rmse"])}
    manifest = {
        "status": "complete", "protocol": "aircraft_ipc3_retarget_compression_v1",
        "dataset": "A_imsize224", "ipc": 3, "images": 300, "classes": 100,
        "source": "exact decoded RandomReal source seed0 IPC3 224x224 files",
        "encoded_storage": "lossless 8-bit PNG; every encoded file is closed and reloaded before decode",
        "decoded_storage": "lossless 8-bit 224x224 PNG used verbatim by FKD and Student",
        "simple_resampling": "PIL Lanczos encode and decode",
        "retarget_encoding": "exact source-row area integration over inverse nonuniform coordinate bins",
        "retarget_decoding": "linear sample of quantized encoded rows at the frozen forward coordinate map; restores R0 row positions/scale",
        "bbox_rule": "official bbox mapped to R0 warp224; subject target density 0.9, background minimum 0.1; subject reduced only when required",
        "attention_rule": "TransFG rollout row mean; reflect Gaussian sigma5 smoothing; capped water-fill density in [0.1,0.9] summing112",
        "isotropic_pixel_budget": 158 * 158, "strip_pixel_budget": 224 * 112,
        "isotropic_minus_strip_fraction": (158 * 158 - 224 * 112) / (224 * 112),
        "bbox_audit": {"width224": stats(r["official_bbox224"][2] - r["official_bbox224"][0] for r in records),
                       "height224": stats(r["official_bbox224"][3] - r["official_bbox224"][1] for r in records),
                       "allocated_integer_rows": stats(bbox_heights), "subject_density": stats(bbox_subject_density),
                       "background_density": stats(bbox_background_density), "reduced_images": bbox_reduced,
                       "reduced_fraction": bbox_reduced / len(records)},
        "attention_density_audit": {"minimum": stats(attention_min), "maximum": stats(attention_max),
                                    "density_entropy_nats": stats(attention_entropy), "floor": 0.1, "ceiling": 0.9,
                                    "smoothing_sigma_rows": 5.0},
        "transfg_checkpoint": str(args.transfg_checkpoint.resolve()),
        "transfg_checkpoint_sha256": sha256_file(args.transfg_checkpoint), "transfg_architecture": architecture,
        "outputs": outputs, "records": output_records,
    }
    atomic_json(manifest, args.output_root / "construction_manifest.json")
    print(json.dumps({"status": manifest["status"], "outputs": outputs,
                      "bbox_audit": manifest["bbox_audit"],
                      "attention_density_audit": manifest["attention_density_audit"]}, indent=2))


if __name__ == "__main__":
    main()
