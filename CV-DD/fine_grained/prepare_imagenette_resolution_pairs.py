"""Select ImageNette images by class-conditional H20 entropy and create A/B pairs."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
IPCS = (10, 50)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if not destination.is_symlink() or destination.resolve() != source.resolve():
            raise RuntimeError(f"selection collision: {destination}")
    else:
        os.symlink(source.resolve(), destination)


def load_teacher(path: Path) -> nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


def canonical_a_tensor(path: Path) -> tuple[torch.Tensor, tuple[int, int], bytes]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        source_size = rgb.size
        a = rgb.resize((224, 224), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    a.save(buffer, format="JPEG")
    encoded = buffer.getvalue()
    with Image.open(io.BytesIO(encoded)) as decoded:
        tensor = transforms.ToTensor()(decoded.convert("RGB"))
    return tensor, source_size, encoded


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "construction_manifest.json"
    if args.skip_completed and manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            print(f"ImageNette A/B construction already complete: {manifest_path}")
            return
    if not args.teacher.is_file():
        raise FileNotFoundError(args.teacher)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    teacher = load_teacher(args.teacher)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    source_root = (args.data_root / "train").resolve()
    classes = sorted(path for path in source_root.iterdir() if path.is_dir())
    if len(classes) != 10:
        raise RuntimeError(f"found {len(classes)} ImageNette classes")
    records = []
    all_logits = []
    all_labels = []
    all_sample_ids = []
    total_images = 0
    for class_id, class_dir in enumerate(classes):
        candidates = sorted(
            path.resolve() for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(candidates) < 50:
            raise RuntimeError(f"{class_dir}: only {len(candidates)} candidates")
        tensors = []
        source_sizes = []
        a_jpegs = []
        for source in candidates:
            tensor, source_size, encoded = canonical_a_tensor(source)
            tensors.append(tensor)
            source_sizes.append(source_size)
            a_jpegs.append(encoded)
        logits_parts = []
        for start in range(0, len(tensors), args.batch_size):
            images = torch.stack(tensors[start : start + args.batch_size])
            images = (images - mean) / std
            logits_parts.append(teacher(images.cuda(non_blocking=True)).cpu().float())
        logits = torch.cat(logits_parts)
        log_probabilities = torch.log_softmax(logits / 20.0, dim=1)
        entropies = -(log_probabilities.exp() * log_probabilities).sum(dim=1)
        sample_ids = [path.relative_to(source_root).as_posix() for path in candidates]
        ranking = sorted(range(len(candidates)), key=lambda index: (float(entropies[index]), sample_ids[index]))
        class_records = []
        for rank, candidate_index in enumerate(ranking[:50]):
            source = candidates[candidate_index]
            a_path = args.output_root / "canonical_a224" / class_dir.name / f"class{class_id:05d}_id{rank:05d}.jpg"
            cache_path = args.output_root / "cache112" / class_dir.name / f"class{class_id:05d}_id{rank:05d}.png"
            b_path = args.output_root / "canonical_b112up" / class_dir.name / f"class{class_id:05d}_id{rank:05d}.jpg"
            a_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            b_path.parent.mkdir(parents=True, exist_ok=True)
            a_path.write_bytes(a_jpegs[candidate_index])
            with Image.open(a_path) as image:
                low = image.convert("RGB").resize((112, 112), Image.Resampling.BILINEAR)
            low.save(cache_path, format="PNG")
            with Image.open(cache_path) as image:
                b = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
            b.save(b_path, format="JPEG")
            class_records.append(
                {
                    "rank": rank, "candidate_index": candidate_index,
                    "sample_id": sample_ids[candidate_index],
                    "source_path": str(source), "source_sha256": sha256(source),
                    "source_size": list(source_sizes[candidate_index]),
                    "entropy_t20": float(entropies[candidate_index]),
                    "a_path": str(a_path.resolve()), "a_sha256": sha256(a_path),
                    "cache112_path": str(cache_path.resolve()), "cache112_sha256": sha256(cache_path),
                    "b_path": str(b_path.resolve()), "b_sha256": sha256(b_path),
                }
            )
        for ipc in IPCS:
            for row in class_records[:ipc]:
                name = f"class{class_id:05d}_id{row['rank']:05d}.jpg"
                link(Path(row["a_path"]), args.output_root / f"selected/a_image/ipc{ipc}" / class_dir.name / name)
                link(Path(row["b_path"]), args.output_root / f"selected/b_image/ipc{ipc}" / class_dir.name / name)
        records.append(
            {"class_id": class_id, "class_folder": class_dir.name,
             "candidate_images": len(candidates), "images": class_records}
        )
        all_logits.append(logits.numpy().astype(np.float32))
        all_labels.append(np.full(len(candidates), class_id, dtype=np.int64))
        all_sample_ids.extend(sample_ids)
        total_images += len(candidates)
        print(f"class {class_id + 1}/10 scored and constructed", flush=True)
    if total_images != 9469:
        raise RuntimeError(f"expected 9469 source images, found {total_images}")
    np.savez_compressed(
        args.output_root / "teacher_logits.npz",
        logits=np.concatenate(all_logits), labels=np.concatenate(all_labels),
        sample_ids=np.asarray(all_sample_ids),
    )
    for arm in ("a_image", "b_image"):
        for ipc in IPCS:
            files = list((args.output_root / f"selected/{arm}/ipc{ipc}").glob("*/*.jpg"))
            if len(files) != 10 * ipc:
                raise RuntimeError(f"{arm} IPC{ipc}: found {len(files)} files")
    payload = {
        "status": "complete", "dataset": "imagenet-nette", "classes": 10,
        "source_root": str(source_root), "source_images": total_images,
        "selection": "per-class entropy of softmax(raw_logits/20), ascending",
        "selection_temperature": 20.0, "selection_log": "natural",
        "selection_teacher_mode": "eval", "selection_ties": "sample_id ascending",
        "selection_view": "RGB -> PIL bilinear warp 224x224 -> Tensor -> ImageNet normalize",
        "teacher_checkpoint": str(args.teacher.resolve()), "teacher_sha256": sha256(args.teacher),
        "teacher_logits": str((args.output_root / "teacher_logits.npz").resolve()),
        "teacher_logits_dtype": "float32", "ipcs": list(IPCS), "nested": True,
        "a_construction": "selected scoring-view RGB, PIL default JPEG at 224x224",
        "shared_cache": "decode A JPEG, PIL bilinear 224->112, lossless PNG",
        "b_construction": "shared 112 PNG, PIL bilinear 112->224, PIL default JPEG",
        "a_b_filename_and_imagefolder_index_aligned": True,
        "class_records": records,
    }
    atomic_json(payload, manifest_path)
    print(json.dumps({key: value for key, value in payload.items() if key != "class_records"}, indent=2))


if __name__ == "__main__":
    main()
