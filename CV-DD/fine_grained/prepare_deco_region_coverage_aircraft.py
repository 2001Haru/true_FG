"""Build controlled random-candidate and joint-coverage DeCO mosaics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as TF

from vit_teacher_common import IMAGENET_MEAN, IMAGENET_STD, atomic_json, build_teacher, sha256_file


def stable_digest(*values):
    return hashlib.sha256("\0".join(map(str, values)).encode("utf-8")).digest()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    overlap = intersection(a, b)
    return overlap / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap)


@torch.inference_mode()
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


def pick_attention_candidates(scores, side, window, count, max_iou, fixed_first):
    ranked = torch.argsort(scores.reshape(-1), descending=True, stable=True).tolist()
    selected = [tuple(fixed_first)]
    for flat in ranked:
        y, x = divmod(flat, side)
        box = (x, y, x + window, y + window)
        if box in selected:
            continue
        if all(iou(box, previous) <= max_iou for previous in selected):
            selected.append(box)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise RuntimeError(f"attention candidate selection returned {len(selected)} windows, expected {count}")
    return selected


def raw_tile(row, box):
    with Image.open(row["raw_path"]) as handle:
        raw = handle.convert("RGB")
    x0, y0, x1, y1 = box
    extent = (x0 / 224 * raw.width, y0 / 224 * raw.height, x1 / 224 * raw.width, y1 / 224 * raw.height)
    return raw.transform((112, 112), Image.Transform.EXTENT, extent, Image.Resampling.BILINEAR), extent


def facility_utility(features, selected):
    if not selected:
        return float("-inf")
    return float((features @ features[selected].T).max(dim=1).values.mean())


def joint_facility_selection(features, source_indices):
    selected, locked = [], set()
    while len(locked) < len(set(source_indices)):
        best = None
        for candidate, source in enumerate(source_indices):
            if source in locked:
                continue
            utility = facility_utility(features, selected + [candidate])
            key = (utility, -source, -candidate)
            if best is None or key > best[0]:
                best = (key, candidate, source)
        selected.append(best[1])
        locked.add(best[2])
    return selected


def refine_facility_selection(features, selected, source_indices):
    selected = list(selected)
    changed = True
    while changed:
        changed = False
        for source in sorted(set(source_indices)):
            position = next(index for index, candidate in enumerate(selected) if source_indices[candidate] == source)
            old = selected[position]
            best = (facility_utility(features, selected), -old, old)
            for candidate, candidate_source in enumerate(source_indices):
                if candidate_source != source:
                    continue
                proposal = selected.copy()
                proposal[position] = candidate
                key = (facility_utility(features, proposal), -candidate, candidate)
                if key > best:
                    best = key
            if best[2] != old:
                selected[position] = best[2]
                changed = True
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-manifest", required=True, type=Path)
    p.add_argument("--transfg-source", required=True, type=Path)
    p.add_argument("--transfg-checkpoint", required=True, type=Path)
    p.add_argument("--dino-model-root", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--selection-seed", default=20260915, type=int)
    p.add_argument("--window", default=119, type=int)
    p.add_argument("--candidates-per-source", default=4, type=int)
    p.add_argument("--candidate-max-iou", default=0.25, type=float)
    p.add_argument("--batch-size", default=24, type=int)
    args = p.parse_args()
    base = json.loads(args.base_manifest.read_text())
    if base.get("status") != "complete" or base.get("regions") != 1200 or base.get("window_size_reference224") != args.window:
        raise RuntimeError("base manifest is not the expected completed DeCO seed0 construction")
    rows = []
    for source_index, old in enumerate(base["regions_detail"]):
        row = {key: old[key] for key in ("class_id", "class", "mosaic", "tile", "image_id", "raw_path", "is_r0_source")}
        row["source_index"] = source_index
        row["base_fg_window224"] = old["fg_window224"]
        rows.append(row)

    teacher, architecture = build_teacher("transfg", args.transfg_source, weights_path=None, checkpoint_path=args.transfg_checkpoint)
    teacher.cuda().eval()
    side = 224 - args.window + 1
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset:offset + args.batch_size]
        images = []
        for row in batch:
            with Image.open(row["raw_path"]) as handle:
                reference = handle.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            images.append(TF.normalize(TF.pil_to_tensor(reference).float().div_(255), IMAGENET_MEAN, IMAGENET_STD))
        maps = rollout(teacher, torch.stack(images).cuda()).cpu()
        pooled = F.avg_pool2d(maps[:, None], args.window, stride=1)[:, 0]
        for row, scores in zip(batch, pooled):
            argmax_flat = int(scores.reshape(-1).argmax())
            argmax_y, argmax_x = divmod(argmax_flat, side)
            recomputed_argmax = [argmax_x, argmax_y, argmax_x + args.window, argmax_y + args.window]
            candidates = pick_attention_candidates(
                scores, side, args.window, args.candidates_per_source, args.candidate_max_iou,
                row["base_fg_window224"],
            )
            row["recomputed_argmax_window224"] = recomputed_argmax
            row["recomputed_argmax_matches_base"] = recomputed_argmax == row["base_fg_window224"]
            row["candidates"] = [{"window224": list(box), "attention_mean": float(scores[box[1], box[0]])} for box in candidates]
    del teacher
    torch.cuda.empty_cache()

    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as error:
        raise RuntimeError("transformers is required for DINOv2 region encoding") from error
    processor = AutoImageProcessor.from_pretrained(str(args.dino_model_root), local_files_only=True)
    model = AutoModel.from_pretrained(str(args.dino_model_root), local_files_only=True).cuda().eval()
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(processor.image_mean, processor.image_std),
    ])
    tiles, candidate_index = [], []
    for row_index, row in enumerate(rows):
        for local_index, candidate in enumerate(row["candidates"]):
            tile, extent = raw_tile(row, candidate["window224"])
            candidate["raw_window"] = list(extent)
            tiles.append(transform(tile))
            candidate_index.append((row_index, local_index))
    features = []
    with torch.inference_mode():
        for offset in range(0, len(tiles), args.batch_size * 2):
            batch = torch.stack(tiles[offset:offset + args.batch_size * 2]).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                encoded = model(pixel_values=batch).last_hidden_state[:, 0]
            features.append(F.normalize(encoded.float(), dim=1).cpu())
    features = torch.cat(features)
    if not torch.isfinite(features).all() or float((features.norm(dim=1) - 1).abs().max()) > 1e-5:
        raise RuntimeError("invalid DINO feature matrix")
    for global_index, (row_index, local_index) in enumerate(candidate_index):
        rows[row_index]["candidates"][local_index]["feature_index"] = global_index

    choices = {"candidate_random": {}, "joint_coverage": {}}
    class_audits = []
    for class_name in sorted({row["class"] for row in rows}):
        class_rows = [row for row in rows if row["class"] == class_name]
        if len(class_rows) != 12:
            raise RuntimeError(f"{class_name} has {len(class_rows)} sources")
        class_indices, source_ids = [], []
        for local_source, row in enumerate(class_rows):
            for candidate in row["candidates"]:
                class_indices.append(candidate["feature_index"])
                source_ids.append(local_source)
            random_local = int.from_bytes(stable_digest("region-candidate-random-v1", args.selection_seed, class_name, row["image_id"])[:8], "little") % len(row["candidates"])
            choices["candidate_random"][row["source_index"]] = random_local
        class_features = features[class_indices]
        current_flat = [source * len(row["candidates"]) for source, row in enumerate(class_rows)]
        random_flat = [source * len(row["candidates"]) + choices["candidate_random"][row["source_index"]] for source, row in enumerate(class_rows)]
        greedy_flat = joint_facility_selection(class_features, source_ids)
        selected_flat = max((current_flat, random_flat, greedy_flat), key=lambda selected: facility_utility(class_features, selected))
        selected_flat = refine_facility_selection(class_features, selected_flat, source_ids)
        if len(selected_flat) != 12 or len({source_ids[index] for index in selected_flat}) != 12:
            raise RuntimeError("joint coverage did not choose exactly one candidate per source")
        for flat in selected_flat:
            row = class_rows[source_ids[flat]]
            choices["joint_coverage"][row["source_index"]] = flat % len(row["candidates"])
        class_audits.append({
            "class": class_name,
            "candidate_count": len(class_indices),
            "current_fg_utility": facility_utility(class_features, current_flat),
            "candidate_random_utility": facility_utility(class_features, random_flat),
            "joint_coverage_utility": facility_utility(class_features, selected_flat),
        })

    output_roots = {name: args.output_root / "selected" / name / "ipc3" for name in choices}
    for root in output_roots.values():
        root.mkdir(parents=True, exist_ok=True)
    grouped = {}
    for row in rows:
        grouped.setdefault((row["class"], row["mosaic"]), []).append(row)
    for (class_name, mosaic), group in sorted(grouped.items()):
        for method, method_choices in choices.items():
            output = Image.new("RGB", (224, 224))
            for row in sorted(group, key=lambda item: item["tile"]):
                selected_index = method_choices[row["source_index"]]
                tile, _ = raw_tile(row, row["candidates"][selected_index]["window224"])
                output.paste(tile, ((row["tile"] % 2) * 112, (row["tile"] // 2) * 112))
                row[method + "_candidate_index"] = selected_index
            destination = output_roots[method] / class_name / f"mosaic_{mosaic}.png"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(".png.tmp")
            output.save(temporary, format="PNG", compress_level=0)
            os.replace(temporary, destination)

    outputs = {}
    for method, root in output_roots.items():
        files = sorted(root.glob("*/*.png"))
        digest = hashlib.sha256()
        for path in files:
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(bytes.fromhex(sha(path)))
        outputs[method] = {"root": str(root.resolve()), "images": len(files), "tree_sha256": digest.hexdigest()}
        if len(files) != 300:
            raise RuntimeError(f"{method} has {len(files)} mosaics")
    summary = {}
    for method in choices:
        picked = [row[method + "_candidate_index"] for row in rows]
        summary[method] = {
            "candidate_index_histogram": {str(index): picked.count(index) for index in range(args.candidates_per_source)},
            "mean_class_facility_utility": sum(row[method + "_utility"] for row in class_audits) / len(class_audits),
            "same_as_current_fg_fraction": sum(index == 0 for index in picked) / len(picked),
        }
    result = {
        "status": "complete", "protocol": "deco_region_joint_coverage_v1", "dataset": "A_imsize224", "ipc": 3,
        "classes": 100, "stored_images": 300, "regions": 1200, "regions_per_class": 12,
        "independent_sources_per_class": 12, "window_size_reference224": args.window,
        "window_area_ratio": args.window ** 2 / 224 ** 2, "tile_size": 112,
        "candidates_per_source": args.candidates_per_source, "candidate_max_iou": args.candidate_max_iou,
        "candidate_rule": "top TransFG attention-mean windows with deterministic NMS; candidate0 exactly reproduces current FG-region",
        "recomputed_argmax_matches_base_fraction": sum(row["recomputed_argmax_matches_base"] for row in rows) / len(rows),
        "dino_geometry": "stored 112x112 RGB tile; bicubic Resize256; CenterCrop224; DINOv2-base CLS; L2 normalization; cosine",
        "coverage_rule": "classwise greedy facility-location over all 48 candidate tiles, followed by deterministic one-source swaps to convergence; maximize mean max cosine similarity with one candidate per source",
        "random_rule": "one uniform stable-SHA256 candidate index per source from the identical candidate pool",
        "selection_seed": args.selection_seed, "base_manifest": str(args.base_manifest.resolve()),
        "base_manifest_sha256": sha256_file(args.base_manifest), "transfg_checkpoint_sha256": sha256_file(args.transfg_checkpoint),
        "architecture": architecture, "outputs": outputs, "selection_summary": summary,
        "class_audits": class_audits, "regions_detail": rows,
    }
    atomic_json(result, args.output_root / "construction_manifest.json")
    print(json.dumps({"status": result["status"], "outputs": outputs, "selection_summary": summary}, indent=2))


if __name__ == "__main__":
    main()
