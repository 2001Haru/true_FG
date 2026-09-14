"""Shared, audited plumbing for the Aircraft ViT/TransFG teacher experiment."""

from __future__ import annotations

import hashlib
import importlib
import json
import random
import sys
import types
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, Sampler
from torchvision import datasets
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PLAIN_COMMIT = "460a162767de1722a014ed2261463dbbc01196b6"
TRANSFG_COMMIT = "9336fba46a4ac8ed2e33072c5f74ab459b114e4a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_u64(*items: object) -> int:
    payload = "\x1f".join(map(str, items)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_commit(source_root: Path) -> str:
    marker = source_root / "PINNED_COMMIT"
    if not marker.is_file():
        raise RuntimeError(f"vendored source lacks PINNED_COMMIT: {source_root}")
    if (source_root / ".git").exists():
        raise RuntimeError(f"vendored source must not contain nested .git metadata: {source_root}")
    return marker.read_text(encoding="utf-8").strip()


def import_upstream(source_root: Path):
    try:
        import ml_collections  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("ml_collections")

        class ConfigDict(dict):
            """Minimal upstream-compatible container for the pinned config files."""

            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError as error:
                    raise AttributeError(name) from error

            def __setattr__(self, name, value):
                self[name] = value

            def __delattr__(self, name):
                try:
                    del self[name]
                except KeyError as error:
                    raise AttributeError(name) from error

        module.ConfigDict = ConfigDict
        module.__version__ = "repository_fallback_v1"
        sys.modules["ml_collections"] = module
    # CV-DD itself may already have imported a top-level package named
    # ``models`` before this helper is called (notably in FKD replay).  The two
    # pinned upstreams also use that generic package name.  A fresh process is
    # used for each teacher kind, so evict only this ambiguous package namespace
    # before importing the explicitly selected vendored implementation.
    for name in list(sys.modules):
        if name == "models" or name.startswith("models."):
            del sys.modules[name]
    sys.path.insert(0, str(source_root.resolve()))
    try:
        module = importlib.import_module("models.modeling")
    finally:
        sys.path.pop(0)
    module_path = Path(module.__file__).resolve()
    if source_root.resolve() not in module_path.parents:
        raise RuntimeError(f"wrong models.modeling imported: {module_path}")
    return module


def build_teacher(
    kind: str,
    source_root: Path,
    *,
    weights_path: Path | None,
    checkpoint_path: Path | None = None,
) -> tuple[nn.Module, dict]:
    if kind not in {"vit", "transfg"}:
        raise ValueError(kind)
    expected_commit = PLAIN_COMMIT if kind == "vit" else TRANSFG_COMMIT
    actual_commit = source_commit(source_root)
    if actual_commit != expected_commit:
        raise RuntimeError(f"{kind} source commit {actual_commit} != {expected_commit}")
    upstream = import_upstream(source_root)
    config = upstream.CONFIGS["ViT-B_16"]
    if kind == "transfg":
        config.split = "overlap"
        config.slide_step = 12
        model = upstream.VisionTransformer(
            config, img_size=224, num_classes=100, smoothing_value=0.0, zero_head=True
        )
    else:
        model = upstream.VisionTransformer(
            config, img_size=224, num_classes=100, zero_head=True, vis=False
        )
    if checkpoint_path is None:
        if weights_path is None:
            raise ValueError("weights_path is required for initialization")
        model.load_from(np.load(weights_path))
        head = model.part_head if kind == "transfg" else model.head
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
    else:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model.load_state_dict(state, strict=True)
    embedding = model.transformer.embeddings
    encoder = model.transformer.encoder
    head = model.part_head if kind == "transfg" else model.head
    position_tokens = int(embedding.position_embeddings.shape[1])
    stride = tuple(int(value) for value in embedding.patch_embeddings.stride)
    ordinary_blocks = len(encoder.layer)
    audit = {
        "kind": kind,
        "source_root": str(source_root.resolve()),
        "source_commit": actual_commit,
        "input_size": 224,
        "patch_size": [16, 16],
        "patch_stride": list(stride),
        "position_tokens": position_tokens,
        "ordinary_encoder_blocks": ordinary_blocks,
        "has_part_layer": hasattr(encoder, "part_layer"),
        "classification_head_weight_norm": float(head.weight.detach().float().norm()),
        "classification_head_bias_norm": float(head.bias.detach().float().norm()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "config_container": getattr(sys.modules["ml_collections"], "__version__", "unknown"),
        "pretrained_block_loading": (
            "encoder_blocks_0_to_10_loaded; part_layer_randomly_initialized"
            if kind == "transfg"
            else "encoder_blocks_0_to_11_loaded"
        ),
    }
    if checkpoint_path is None:
        with np.load(weights_path) as weights:
            def upstream_query(block_index: int) -> torch.Tensor:
                key = f"Transformer/encoderblock_{block_index}/MultiHeadDotProductAttention_1/query/kernel"
                return torch.from_numpy(weights[key]).view(768, 768).t().float()

            if kind == "transfg":
                loaded = encoder.layer[10].attn.query.weight.detach().cpu().float()
                random_part = encoder.part_layer.attn.query.weight.detach().cpu().float()
                audit["loaded_block10_query_max_abs_difference"] = float(
                    (loaded - upstream_query(10)).abs().max()
                )
                audit["random_part_vs_pretrained_block11_query_rms_difference"] = float(
                    (random_part - upstream_query(11)).square().mean().sqrt()
                )
                if audit["loaded_block10_query_max_abs_difference"] != 0.0:
                    raise RuntimeError("TransFG ordinary block 10 did not load exactly")
                if audit["random_part_vs_pretrained_block11_query_rms_difference"] < 1e-6:
                    raise RuntimeError("TransFG part layer unexpectedly matches pretrained block 11")
            else:
                loaded = encoder.layer[11].attn.query.weight.detach().cpu().float()
                audit["loaded_block11_query_max_abs_difference"] = float(
                    (loaded - upstream_query(11)).abs().max()
                )
                if audit["loaded_block11_query_max_abs_difference"] != 0.0:
                    raise RuntimeError("plain ViT block 11 did not load exactly")
    expected = (
        {"stride": (12, 12), "tokens": 325, "blocks": 11, "part": True}
        if kind == "transfg"
        else {"stride": (16, 16), "tokens": 197, "blocks": 12, "part": False}
    )
    observed = {
        "stride": stride,
        "tokens": position_tokens,
        "blocks": ordinary_blocks,
        "part": hasattr(encoder, "part_layer"),
    }
    if observed != expected:
        raise RuntimeError(f"architecture mismatch: {observed} != {expected}")
    if checkpoint_path is None and (
        audit["classification_head_weight_norm"] != 0.0
        or audit["classification_head_bias_norm"] != 0.0
    ):
        raise RuntimeError("classification head was not explicitly zeroed")
    return model, audit


def teacher_logits_and_loss(
    model: nn.Module, kind: str, images: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if kind == "transfg":
        tokens = model.transformer(images)
        features = tokens[:, 0].float()
        logits = model.part_head(tokens[:, 0])
        normalized = F.normalize(features, dim=1)
        cosine = normalized @ normalized.t()
        positive = labels[:, None].eq(labels[None, :]).float()
        negative = 1.0 - positive
        contrast = ((1.0 - cosine) * positive).sum()
        contrast = contrast + ((cosine - 0.4).clamp_min(0.0) * negative).sum()
        contrast = contrast / float(labels.numel() ** 2)
    else:
        logits = model(images)[0]
        contrast = logits.float().new_zeros(())
    classification = F.cross_entropy(logits.float(), labels, reduction="mean")
    loss = classification + contrast
    parts = {"classification": float(classification.detach()), "contrast": float(contrast.detach())}
    return logits, loss, parts


def inference_logits(model: nn.Module, kind: str, images: torch.Tensor) -> torch.Tensor:
    output = model(images)
    return output if kind == "transfg" else output[0]


class AircraftRawTrain(Dataset):
    """Use the current class/index tree but load each identity from original pixels."""

    def __init__(self, index_root: Path, raw_image_root: Path, seed: int):
        index = datasets.ImageFolder(index_root)
        self.classes = index.classes
        self.class_to_idx = index.class_to_idx
        self.samples: list[tuple[Path, int, str]] = []
        for indexed_path, target in index.samples:
            identity = Path(indexed_path).name
            raw_path = raw_image_root / identity
            if not raw_path.is_file():
                raise FileNotFoundError(raw_path)
            self.samples.append((raw_path, target, identity))
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, key: tuple[int, int, int]):
        index, update, slot = key
        path, target, identity = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
        rng = random.Random(stable_u64("aircraft-vit-augmentation-v1", self.seed, update, slot, identity))
        top = rng.randrange(33)
        left = rng.randrange(33)
        image = image.crop((left, top, left + 224, top + 224))
        if rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        tensor = TF.pil_to_tensor(image).float().div_(255.0)
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        return tensor, target


class StandardAircraftTest(Dataset):
    def __init__(self, root: Path):
        self.index = datasets.ImageFolder(root)
        self.classes = self.index.classes

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int):
        path, target = self.index.samples[index]
        with Image.open(path) as source:
            image = source.convert("RGB")
        if image.size != (224, 224):
            raise RuntimeError(f"standard test image is not 224x224: {path} {image.size}")
        tensor = TF.pil_to_tensor(image).float().div_(255.0)
        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        return tensor, target


class FixedUpdateBatchSampler(Sampler[list[tuple[int, int, int]]]):
    """A deterministic, model-independent stream of true batch-64 updates."""

    def __init__(self, size: int, batch_size: int, total_updates: int, start_update: int, seed: int):
        self.size = size
        self.batch_size = batch_size
        self.total_updates = total_updates
        self.start_update = start_update
        self.seed = seed

    def __len__(self) -> int:
        return self.total_updates - self.start_update

    def __iter__(self) -> Iterator[list[tuple[int, int, int]]]:
        generator = torch.Generator().manual_seed(self.seed)
        update = 0
        while update < self.total_updates:
            permutation = torch.randperm(self.size, generator=generator).tolist()
            usable = self.size - (self.size % self.batch_size)
            for offset in range(0, usable, self.batch_size):
                if update >= self.total_updates:
                    return
                if update >= self.start_update:
                    yield [
                        (index, update, slot)
                        for slot, index in enumerate(permutation[offset : offset + self.batch_size])
                    ]
                update += 1


def class_mapping_digest(classes: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(classes).encode("utf-8")).hexdigest()


def learning_rate(update_number: int, total: int = 10_000, warmup: int = 500, peak: float = 0.03) -> float:
    if not 1 <= update_number <= total:
        raise ValueError(update_number)
    if update_number <= warmup:
        return peak * update_number / warmup
    progress = (update_number - warmup) / (total - warmup)
    return peak * 0.5 * (1.0 + np.cos(np.pi * progress))
