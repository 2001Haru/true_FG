"""Deterministic released-RDED ImageNette IPC10/50 image construction."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IPCS = (10, 50)
CLASSES = 10
CANDIDATES = 300
NUM_CROPS = 5
FACTOR = 2
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest(namespace: str, seed: int, identity: str) -> bytes:
    return hashlib.sha256(
        f"rded-imagenette-v1\0{namespace}\0{seed}\0{identity}".encode("utf-8")
    ).digest()


def stable_seed(namespace: str, seed: int, identity: str) -> int:
    return int.from_bytes(digest(namespace, seed, identity)[:8], "big") % (2**63 - 1)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_teacher(path: Path) -> nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, CLASSES)
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


def deterministic_crop(image: torch.Tensor, seed: int) -> tuple[torch.Tensor, list[int]]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        top, left, height, width = transforms.RandomResizedCrop.get_params(
            image, scale=(0.08, 1.0), ratio=(1.0, 1.0)
        )
    crop = TF.resized_crop(
        image, top, left, height, width, [112, 112],
        interpolation=InterpolationMode.BILINEAR, antialias=True,
    )
    return crop, [int(top), int(left), int(height), int(width)]


def center_pad(images: torch.Tensor) -> torch.Tensor:
    return F.pad(images, (56, 56, 56, 56))


def rank_groups(ipc: int) -> list[list[int]]:
    return [[quadrant * ipc + image_id for quadrant in range(4)] for image_id in range(ipc)]


def make_mosaics(selected: torch.Tensor, ipc: int) -> torch.Tensor:
    if selected.shape != (ipc * 4, 3, 112, 112):
        raise RuntimeError(f"unexpected selected shape: {tuple(selected.shape)}")
    mosaics = torch.empty((ipc, 3, 224, 224), dtype=selected.dtype)
    for image_id, ranks in enumerate(rank_groups(ipc)):
        for quadrant, rank in enumerate(ranks):
            row, column = divmod(quadrant, 2)
            mosaics[image_id, :, row * 112:(row + 1) * 112, column * 112:(column + 1) * 112] = selected[rank]
    return mosaics


@torch.no_grad()
def score_class(model, class_id, paths, source_root, seed, forward_batch):
    to_tensor = transforms.ToTensor()
    crops_by_candidate = []
    params_by_candidate = []
    source_sizes = []
    for path in paths:
        relative = path.relative_to(source_root).as_posix()
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            source_sizes.append(rgb.size)
            tensor = to_tensor(rgb)
        crops = []
        parameters = []
        for crop_id in range(NUM_CROPS):
            crop, parameters_one = deterministic_crop(
                tensor, stable_seed("crop", seed, f"{relative}\0{crop_id}")
            )
            crops.append(crop)
            parameters.append(parameters_one)
        crops_by_candidate.append(torch.stack(crops))
        params_by_candidate.append(parameters)
    raw = torch.stack(crops_by_candidate)  # candidate,crop,C,112,112
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    flattened = raw.permute(1, 0, 2, 3, 4).reshape(-1, 3, 112, 112)
    logits = []
    for start in range(0, flattened.shape[0], forward_batch):
        batch = (flattened[start:start + forward_batch] - mean) / std
        logits.append(model(center_pad(batch).cuda(non_blocking=True)).cpu())
    labels = torch.full((flattened.shape[0],), class_id, dtype=torch.long)
    losses = F.cross_entropy(torch.cat(logits), labels, reduction="none").reshape(NUM_CROPS, CANDIDATES)
    best_crop = losses.argmin(dim=0)
    candidate_ids = torch.arange(CANDIDATES)
    best_losses = losses[best_crop, candidate_ids]
    crop_major = raw.permute(1, 0, 2, 3, 4)
    best_images = crop_major[best_crop, candidate_ids]
    ranking = torch.argsort(best_losses, stable=True)
    ranked_images = best_images[ranking]
    records = []
    for rank, candidate_tensor in enumerate(ranking):
        candidate = int(candidate_tensor)
        crop_id = int(best_crop[candidate])
        path = paths[candidate]
        records.append(
            {
                "rank": rank, "candidate_index": candidate,
                "sample_id": path.relative_to(source_root).as_posix(),
                "source_path": str(path), "source_sha256": sha256(path),
                "source_size": list(source_sizes[candidate]),
                "best_crop_id": crop_id,
                "best_crop_tlhw": params_by_candidate[candidate][crop_id],
                "teacher_cross_entropy": float(best_losses[candidate]),
                "teacher_true_probability": float(math.exp(-float(best_losses[candidate]))),
            }
        )
    return ranked_images, records


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--generation-seed", default=42, type=int)
    parser.add_argument("--forward-batch-size", default=300, type=int)
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    manifest_path = args.output_root / "generation_manifest.json"
    if args.skip_completed and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "complete":
            print(f"RDED ImageNette generation already complete: {manifest_path}")
            return
    torch.manual_seed(args.generation_seed)
    torch.cuda.manual_seed_all(args.generation_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    teacher = load_teacher(args.teacher)
    source_root = (args.data_root / "train").resolve()
    classes = sorted(path for path in source_root.iterdir() if path.is_dir())
    if len(classes) != CLASSES:
        raise RuntimeError(f"found {len(classes)} classes")
    class_records = []
    outputs = {str(ipc): [] for ipc in IPCS}
    for class_id, class_dir in enumerate(classes):
        all_paths = sorted(
            path.resolve() for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        paths = sorted(
            all_paths,
            key=lambda path: digest(
                "candidate-order", args.generation_seed,
                path.relative_to(source_root).as_posix(),
            ),
        )[:CANDIDATES]
        ranked, ranking = score_class(
            teacher, class_id, paths, source_root,
            args.generation_seed, args.forward_batch_size,
        )
        class_records.append(
            {"class_id": class_id, "class_folder": class_dir.name,
             "full_class_images": len(all_paths), "ranked_candidates": ranking}
        )
        for ipc in IPCS:
            mosaics = make_mosaics(ranked[:ipc * 4], ipc)
            output_class = args.output_root / f"ipc{ipc}" / class_dir.name
            output_class.mkdir(parents=True, exist_ok=True)
            for image_id, ranks in enumerate(rank_groups(ipc)):
                output = output_class / f"class{class_id:05d}_id{image_id:05d}.jpg"
                # Match the released writer: multiplication followed by uint8
                # conversion (truncation), not round-to-nearest.
                array = (mosaics[image_id].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(array).save(output, format="JPEG")
                outputs[str(ipc)].append(
                    {"class_id": class_id, "class_folder": class_dir.name,
                     "image_id": image_id, "quadrant_source_ranks": ranks,
                     "quadrants": [ranking[rank] for rank in ranks],
                     "output_path": str(output.resolve()), "output_sha256": sha256(output)}
                )
        print(f"class {class_id + 1}/10 complete", flush=True)
    for ipc in IPCS:
        if len(list((args.output_root / f"ipc{ipc}").glob("*/*.jpg"))) != CLASSES * ipc:
            raise RuntimeError(f"IPC{ipc} output count mismatch")
    payload = {
        "status": "complete", "method": "released_rded_joint_topk_mosaic",
        "dataset": "imagenet-nette", "generation_seed": args.generation_seed,
        "source_root": str(source_root), "source_pre_crop_resize": "none; decode original resolution RGB",
        "candidate_selection": "stable SHA256 seeded uniform permutation per class",
        "candidate_count_per_class": CANDIDATES, "num_crops_per_source": NUM_CROPS,
        "crop": {"output_size": [112, 112], "scale": [0.08, 1.0], "ratio": [1.0, 1.0], "interpolation": "bilinear", "antialias": True},
        "scoring": "normalize 112 crop with ImageNet statistics, center zero-pad to 224, Teacher true-class CE",
        "teacher_mode": "eval", "teacher_checkpoint": str(args.teacher.resolve()),
        "teacher_sha256": sha256(args.teacher), "ipcs": list(IPCS), "factor": FACTOR,
        "joint_topk": {"10": 40, "50": 200},
        "ipc_construction": "same candidates/crops/ranking; separately regrouped by released rank mapping",
        "pixel_optimization": False, "class_records": class_records, "outputs": outputs,
    }
    atomic_json(payload, manifest_path)
    print(json.dumps({key: value for key, value in payload.items() if key not in ("class_records", "outputs")}, indent=2))


if __name__ == "__main__":
    main()
