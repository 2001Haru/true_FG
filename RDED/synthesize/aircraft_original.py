"""Deterministic Aircraft adaptation of the released RDED image constructor.

The method remains selection-only: one best crop per source image, joint
Top-(IPC * factor^2) selection, and factor-by-factor mosaicing.  There is no
pixel optimization or BN matching in this program.
"""

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AIRCRAFT_MEAN = (0.4865, 0.5177, 0.5425)
AIRCRAFT_STD = (0.2124, 0.2051, 0.2375)
IPCS = (1, 3, 5)
FACTOR = 2
NUM_CROPS = 5
CANDIDATES_PER_CLASS = 66
CLASSES = 100


def atomic_json_dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(namespace: str, seed: int, identity: str) -> bytes:
    return hashlib.sha256(f"rded-aircraft-v1\0{namespace}\0{seed}\0{identity}".encode("utf-8")).digest()


def stable_seed(namespace: str, seed: int, identity: str) -> int:
    return int.from_bytes(stable_digest(namespace, seed, identity)[:8], "big") % (2**63 - 1)


def candidate_order(paths: list[Path], source_root: Path, seed: int) -> list[Path]:
    return sorted(
        paths,
        key=lambda path: stable_digest(
            "candidate-order", seed, path.relative_to(source_root).as_posix()
        ),
    )


def deterministic_crop(image: torch.Tensor, seed: int) -> tuple[torch.Tensor, list[int]]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        top, left, height, width = transforms.RandomResizedCrop.get_params(
            image, scale=(0.08, 1.0), ratio=(1.0, 1.0)
        )
    crop = TF.resized_crop(
        image,
        top,
        left,
        height,
        width,
        size=[112, 112],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    return crop, [int(top), int(left), int(height), int(width)]


def center_pad(images: torch.Tensor, output_size: int = 224) -> torch.Tensor:
    vertical = output_size - images.shape[-2]
    horizontal = output_size - images.shape[-1]
    return F.pad(
        images,
        (
            horizontal // 2,
            horizontal - horizontal // 2,
            vertical // 2,
            vertical - vertical // 2,
        ),
    )


def mosaic_rank_groups(ipc: int, factor: int = FACTOR) -> list[list[int]]:
    """Return the exact rank assignment used by released RDED mix_images."""
    return [[quadrant * ipc + image_id for quadrant in range(factor**2)] for image_id in range(ipc)]


def make_mosaics(selected: torch.Tensor, ipc: int, factor: int = FACTOR) -> torch.Tensor:
    expected = ipc * factor**2
    if selected.shape[0] != expected:
        raise ValueError(f"Expected {expected} selected crops, got {selected.shape[0]}")
    output_size = selected.shape[-1] * factor
    patch_size = output_size // factor
    mosaics = torch.zeros((ipc, 3, output_size, output_size), dtype=selected.dtype)
    for image_id, ranks in enumerate(mosaic_rank_groups(ipc, factor)):
        for quadrant, rank in enumerate(ranks):
            row, column = divmod(quadrant, factor)
            part = F.interpolate(selected[rank : rank + 1], size=(patch_size, patch_size))[0]
            mosaics[
                image_id,
                :,
                row * patch_size : (row + 1) * patch_size,
                column * patch_size : (column + 1) * patch_size,
            ] = part
    return mosaics


def load_teacher(path: Path) -> nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "model" in payload and isinstance(payload["model"], dict):
        payload = payload["model"]
    if any(key.startswith("module.") for key in payload):
        payload = {key.removeprefix("module."): value for key, value in payload.items()}
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, CLASSES)
    model.load_state_dict(payload, strict=True)
    return model.cuda().eval()


@torch.no_grad()
def score_class(
    model: nn.Module,
    class_id: int,
    class_dir: Path,
    source_root: Path,
    generation_seed: int,
    forward_batch_size: int,
) -> tuple[torch.Tensor, list[dict]]:
    paths = sorted(
        path.resolve()
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(paths) < CANDIDATES_PER_CLASS:
        raise RuntimeError(f"{class_dir}: {len(paths)} images, need {CANDIDATES_PER_CLASS}")
    paths = candidate_order(paths, source_root, generation_seed)[:CANDIDATES_PER_CLASS]
    to_tensor = transforms.ToTensor()
    crops_by_candidate = []
    crop_params_by_candidate = []
    for path in paths:
        relative = path.relative_to(source_root).as_posix()
        with Image.open(path) as image:
            tensor = to_tensor(image.convert("RGB"))
        crops = []
        parameters = []
        for crop_id in range(NUM_CROPS):
            crop, params = deterministic_crop(
                tensor, stable_seed("crop", generation_seed, f"{relative}\0{crop_id}")
            )
            crops.append(crop)
            parameters.append(params)
        crops_by_candidate.append(torch.stack(crops))
        crop_params_by_candidate.append(parameters)

    raw = torch.stack(crops_by_candidate)  # candidate, crop, C, 112, 112
    mean = torch.tensor(AIRCRAFT_MEAN).view(1, 1, 3, 1, 1)
    std = torch.tensor(AIRCRAFT_STD).view(1, 1, 3, 1, 1)
    normalized = (raw - mean) / std
    flattened = normalized.permute(1, 0, 2, 3, 4).reshape(-1, 3, 112, 112)
    logits = []
    padded = center_pad(flattened)
    for start in range(0, padded.shape[0], forward_batch_size):
        logits.append(model(padded[start : start + forward_batch_size].cuda(non_blocking=True)).cpu())
    labels = torch.full((flattened.shape[0],), class_id, dtype=torch.long)
    losses = F.cross_entropy(torch.cat(logits), labels, reduction="none").reshape(
        NUM_CROPS, CANDIDATES_PER_CLASS
    )
    best_crop_ids = losses.argmin(dim=0)
    candidate_ids = torch.arange(CANDIDATES_PER_CLASS)
    best_losses = losses[best_crop_ids, candidate_ids]
    raw_crop_major = raw.permute(1, 0, 2, 3, 4)
    best_crops = raw_crop_major[best_crop_ids, candidate_ids]
    ranking = torch.argsort(best_losses, stable=True)
    ranked_crops = best_crops[ranking]
    records = []
    for rank, candidate_id_tensor in enumerate(ranking):
        candidate_id = int(candidate_id_tensor)
        crop_id = int(best_crop_ids[candidate_id])
        path = paths[candidate_id]
        records.append(
            {
                "rank": rank,
                "candidate_index": candidate_id,
                "source_path": str(path),
                "source_relative_path": path.relative_to(source_root).as_posix(),
                "source_sha256": sha256(path),
                "best_crop_id": crop_id,
                "best_crop_tlhw": crop_params_by_candidate[candidate_id][crop_id],
                "teacher_cross_entropy": float(best_losses[candidate_id]),
                "teacher_true_class_probability": float(math.exp(-float(best_losses[candidate_id]))),
            }
        )
    return ranked_crops, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--generation-seed", required=True, type=int, choices=(42, 43, 44))
    parser.add_argument("--forward-batch-size", default=132, type=int)
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()

    source_root = (args.data_dir / "train").resolve()
    class_dirs = sorted(path for path in source_root.iterdir() if path.is_dir())
    if len(class_dirs) != CLASSES:
        raise RuntimeError(f"Found {len(class_dirs)} classes, expected {CLASSES}")
    if not args.teacher.is_file():
        raise FileNotFoundError(args.teacher)
    output_root = args.output_root.resolve()
    manifest_path = output_root / "generation_manifest.json"
    expected_images = CLASSES * sum(IPCS)
    if args.skip_completed and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = list(output_root.glob("ipc*/*/*.jpg"))
        if manifest.get("status") == "complete" and len(files) == expected_images:
            print(f"RDED generation already complete: {manifest_path}", flush=True)
            return

    random.seed(args.generation_seed)
    np.random.seed(args.generation_seed)
    torch.manual_seed(args.generation_seed)
    torch.cuda.manual_seed_all(args.generation_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = load_teacher(args.teacher)
    class_records = []
    output_records = {str(ipc): [] for ipc in IPCS}

    for class_id, class_dir in enumerate(class_dirs):
        ranked_crops, ranking = score_class(
            model,
            class_id,
            class_dir,
            source_root,
            args.generation_seed,
            args.forward_batch_size,
        )
        class_records.append(
            {
                "class_id": class_id,
                "class_folder": class_dir.name,
                "ranked_candidates": ranking,
            }
        )
        for ipc in IPCS:
            selected = ranked_crops[: ipc * FACTOR**2]
            mosaics = make_mosaics(selected, ipc)
            destination_class = output_root / f"ipc{ipc}" / class_dir.name
            destination_class.mkdir(parents=True, exist_ok=True)
            for image_id, ranks in enumerate(mosaic_rank_groups(ipc)):
                output = destination_class / f"class{class_id:05d}_id{image_id:05d}.jpg"
                array = (mosaics[image_id].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
                Image.fromarray(array).save(output)
                output_records[str(ipc)].append(
                    {
                        "class_id": class_id,
                        "class_folder": class_dir.name,
                        "image_id": image_id,
                        "output_path": str(output),
                        "output_sha256": sha256(output),
                        "quadrant_source_ranks": ranks,
                        "quadrants": [ranking[rank] for rank in ranks],
                    }
                )
        print(f"class {class_id + 1}/{CLASSES} complete", flush=True)

    for ipc in IPCS:
        files = list((output_root / f"ipc{ipc}").glob("*/*.jpg"))
        if len(files) != CLASSES * ipc:
            raise RuntimeError(f"IPC{ipc}: found {len(files)} images")
    manifest = {
        "status": "complete",
        "method": "released_rded_joint_topk_mosaic",
        "adaptation": "Aircraft generation only; downstream evaluation is external standard protocol v2",
        "dataset": "A_imsize224",
        "classes": CLASSES,
        "generation_seed": args.generation_seed,
        "source_root": str(source_root),
        "teacher_checkpoint": str(args.teacher.resolve()),
        "teacher_sha256": sha256(args.teacher),
        "teacher_mode": "eval",
        "teacher_architecture": "torchvision_resnet18",
        "candidate_selection": "stable SHA256 permutation per class, equivalent to seeded uniform shuffle",
        "candidate_count_per_class": CANDIDATES_PER_CLASS,
        "num_crops_per_source": NUM_CROPS,
        "crop": {"output_size": 112, "scale": [0.08, 1.0], "ratio": [1.0, 1.0], "interpolation": "bilinear", "antialias": True},
        "teacher_normalization": {"mean": list(AIRCRAFT_MEAN), "std": list(AIRCRAFT_STD)},
        "scoring": "one minimum-CE crop per source, then joint class Top-(IPC*4)",
        "factor": FACTOR,
        "ipcs": list(IPCS),
        "mosaic_interpolation": "torch.nn.functional.interpolate default nearest",
        "jpeg_save_options": "PIL defaults, matching released RDED",
        "pixel_optimization": False,
        "bn_matching": False,
        "ipc_construction": "same ranked candidates reused, but mosaics independently regrouped for IPC1/3/5",
        "classes_detail": class_records,
        "outputs": output_records,
    }
    atomic_json_dump(manifest, manifest_path)
    print(json.dumps({key: value for key, value in manifest.items() if key not in {"classes_detail", "outputs"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
