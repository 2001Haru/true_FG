"""Frozen utilities for the ImageNette entropy-biased IPC10 experiment."""

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import datasets
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


CLASSES = 10
TRAIN_IMAGES = 9469
TEST_IMAGES = 3925
IPC = 10
TRAIN_SIZE = CLASSES * IPC
IMAGE_SIZE = 256
TRAIN_EPOCHS = 2000
TOTAL_UPDATES = 4000
BATCH_SIZE = 64
EVAL_EVERY_EPOCHS = 20
LR = 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
MILESTONES = (1333, 1666)
LAMBDAS = (-4, -2, 0, 2, 4)
ENDPOINT_LAMBDAS = (-4, 0, 4)
MIDDLE_LAMBDAS = (-2, 2)
SELECTION_SEEDS = (0, 1, 2)
STUDENT_SEEDS = (42, 43, 44)
SELECTION_VIEWS = 16
CALIBRATION_VIEW_SEED = 2026090601
HOLDOUT_VIEW_SEED = 2026090602
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state_dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def identity_seed(*parts) -> int:
    digest = hashlib.sha256("\0".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def gumbel_noise(selection_seed: int, relative_path: str) -> float:
    raw = identity_seed("imagenette-entropy-gumbel-v1", selection_seed, relative_path)
    uniform = (raw + 0.5) / float(1 << 63)
    uniform = min(max(uniform, np.finfo(float).tiny), 1.0 - np.finfo(float).eps)
    return -math.log(-math.log(uniform))


def average_tie_percentiles(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average_rank = (start + stop - 1) / 2.0
        for position in order[start:stop]:
            ranks[position] = average_rank
        start = stop
    denominator = max(len(values) - 1, 1)
    return [rank / denominator for rank in ranks]


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return float(torch.empty((), dtype=torch.float64).uniform_(low, high, generator=generator))


def _rrc_params(image: torch.Tensor, generator: torch.Generator):
    height, width = image.shape[-2:]
    area = height * width
    log_ratio = (math.log(3 / 4), math.log(4 / 3))
    for _ in range(10):
        target_area = area * _uniform(generator, 0.5, 1.0)
        aspect_ratio = math.exp(_uniform(generator, *log_ratio))
        crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
        crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            top = int(torch.randint(0, height - crop_height + 1, (), generator=generator))
            left = int(torch.randint(0, width - crop_width + 1, (), generator=generator))
            return top, left, crop_height, crop_width
    in_ratio = width / height
    if in_ratio < 3 / 4:
        crop_width = width
        crop_height = int(round(crop_width / (3 / 4)))
    elif in_ratio > 4 / 3:
        crop_height = height
        crop_width = int(round(crop_height * (4 / 3)))
    else:
        crop_width, crop_height = width, height
    return (height - crop_height) // 2, (width - crop_width) // 2, crop_height, crop_width


class DeterministicReleasedView:
    """CoDA/Minimax augmentation distribution with identity-keyed randomness."""

    lighting_eigenvalues = torch.tensor([0.2175, 0.0188, 0.0045]).view(1, 3)
    lighting_eigenvectors = torch.tensor(
        [
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ]
    )

    def __call__(self, image: Image.Image, seed: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        image = image.convert("RGB")
        image = TF.resize(
            image,
            IMAGE_SIZE,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        image = TF.center_crop(image, [IMAGE_SIZE, IMAGE_SIZE])
        tensor = TF.to_tensor(image)
        if tensor.dtype != torch.float32 or tensor.min() < 0 or tensor.max() > 1:
            raise RuntimeError("pre-RRC image must be RGB float32 Tensor in [0,1]")
        top, left, height, width = _rrc_params(tensor, generator)
        tensor = TF.resized_crop(
            tensor,
            top,
            left,
            height,
            width,
            [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        if float(torch.rand((), generator=generator)) < 0.5:
            tensor = TF.hflip(tensor)
        factors = {
            0: _uniform(generator, 0.6, 1.4),
            1: _uniform(generator, 0.6, 1.4),
            2: _uniform(generator, 0.6, 1.4),
        }
        for operation in torch.randperm(4, generator=generator).tolist():
            if operation == 0:
                tensor = TF.adjust_brightness(tensor, factors[0])
            elif operation == 1:
                tensor = TF.adjust_contrast(tensor, factors[1])
            elif operation == 2:
                tensor = TF.adjust_saturation(tensor, factors[2])
        alpha = torch.randn(3, generator=generator) * 0.1
        rgb = (
            self.lighting_eigenvectors
            * alpha.view(1, 3)
            * self.lighting_eigenvalues
        ).sum(dim=1)
        return tensor + rgb.view(3, 1, 1)


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def test_transform(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    image = TF.resize(
        image,
        IMAGE_SIZE,
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    image = TF.center_crop(image, [IMAGE_SIZE, IMAGE_SIZE])
    tensor = TF.to_tensor(image)
    if tensor.dtype != torch.float32 or tensor.min() < 0 or tensor.max() > 1:
        raise RuntimeError("test image must be RGB float32 Tensor in [0,1]")
    return normalize(tensor.unsqueeze(0)).squeeze(0)


def load_state_dict_payload(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "ResNet18"):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise RuntimeError(f"unsupported checkpoint payload: {path}")
    cleaned = {}
    for name, value in payload.items():
        while name.startswith("module."):
            name = name.removeprefix("module.")
        cleaned[name] = value
    return cleaned


def build_teacher(checkpoint: Path) -> nn.Module:
    from torchvision import models

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, CLASSES)
    model.load_state_dict(load_state_dict_payload(checkpoint), strict=True)
    model.eval().requires_grad_(False)
    return model


def build_student() -> nn.Module:
    repository_root = Path(__file__).resolve().parents[2]
    module_path = repository_root / "CoDA" / "test" / "resnet_ap.py"
    spec = importlib.util.spec_from_file_location("frozen_coda_resnet_ap", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import ResNetAP from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module.ResNetAP(
        "imagenet",
        depth=10,
        num_classes=CLASSES,
        width=1.0,
        norm_type="instance",
        size=IMAGE_SIZE,
        nch=3,
    )


def validate_official_split(data_root: Path):
    train = datasets.ImageFolder(data_root / "train")
    test = datasets.ImageFolder(data_root / "test")
    if train.classes != test.classes or len(train.classes) != CLASSES:
        raise RuntimeError("ImageNette train/test class mapping mismatch")
    if len(train) != TRAIN_IMAGES or len(test) != TEST_IMAGES:
        raise RuntimeError(
            f"ImageNette image counts are {len(train)}/{len(test)}, expected "
            f"{TRAIN_IMAGES}/{TEST_IMAGES}"
        )
    return train, test


def lr_for_epoch(epoch: int) -> float:
    if not 1 <= epoch <= TRAIN_EPOCHS:
        raise ValueError(epoch)
    if epoch <= MILESTONES[0]:
        return LR
    if epoch <= MILESTONES[1]:
        return LR * 0.2
    return LR * 0.04
