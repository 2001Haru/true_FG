"""Audited two-source-per-parent datasets for the Aircraft IPC3 packing experiment."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


AIRCRAFT_MEAN = (0.4865, 0.5177, 0.5425)
AIRCRAFT_STD = (0.2124, 0.2051, 0.2375)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def stable_u64(*items):
    payload = "\x1f".join(map(str, items)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def decode_vertical(encoded, density):
    source = np.asarray(encoded, dtype=np.uint8).astype(np.float64)
    mapping = np.concatenate(([0.0], np.cumsum(np.asarray(density, dtype=np.float64))))
    if source.shape != (112, 224, 3) or abs(mapping[-1] - 112.0) > 1e-5:
        raise RuntimeError((source.shape, mapping[-1]))
    centers = np.interp(np.arange(224, dtype=np.float64) + 0.5, np.arange(225), mapping) - 0.5
    centers = np.clip(centers, 0.0, 111.0)
    low = np.floor(centers).astype(int); high = np.minimum(low + 1, 111)
    alpha = (centers - low)[:, None, None]
    decoded = source[low] * (1.0 - alpha) + source[high] * alpha
    return Image.fromarray(np.rint(decoded).clip(0, 255).astype(np.uint8), "RGB")


class PairedSourceIndex:
    def __init__(self, manifest_path, image_mode):
        self.manifest_path = Path(manifest_path).resolve()
        payload = json.loads(self.manifest_path.read_text())
        if payload.get("status") != "complete" or payload.get("parents") != 300:
            raise RuntimeError(f"invalid paired source manifest: {manifest_path}")
        if image_mode not in ("reference", "compressed", "uniform"):
            raise ValueError(image_mode)
        self.image_mode = image_mode
        self.rows = sorted(payload["records"], key=lambda row: row["parent_index"])
        if [row["parent_index"] for row in self.rows] != list(range(300)):
            raise RuntimeError("parent indices are not contiguous")
        self.classes = payload["classes"]
        self.targets = [int(row["class_id"]) for row in self.rows]
        self.schedule_seed = int(payload["schedule_seed"])
        self.manifest = payload

    def source_index(self, parent_index, epoch):
        offset = stable_u64("paired-source-offset-v1", self.schedule_seed, parent_index) % 2
        return int((offset + epoch) % 2)

    def load(self, parent_index, source_index):
        row = self.rows[parent_index]
        source = row["sources"][source_index]
        if self.image_mode == "reference":
            with Image.open(source["reference_path"]) as handle:
                image = handle.convert("RGB")
            if image.size != (224, 224):
                raise RuntimeError(f"reference is not 224: {source['reference_path']}")
            return image
        if self.image_mode == "uniform":
            with Image.open(row["uniform_packed_path"]) as handle:
                packed = handle.convert("RGB")
            strip = packed.crop((0, source_index * 112, 224, (source_index + 1) * 112))
            return strip.resize((224, 224), Image.Resampling.LANCZOS)
        with Image.open(row["packed_path"]) as handle:
            packed = handle.convert("RGB")
        if packed.size != (224, 224):
            raise RuntimeError(f"packed image is not 224: {row['packed_path']}")
        strip = packed.crop((0, source_index * 112, 224, (source_index + 1) * 112))
        return decode_vertical(strip, source["density"])


class PairedEpochSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = int(seed)

    def __len__(self):
        return len(self.dataset)

    def permutation(self, epoch):
        generator = torch.Generator().manual_seed(stable_u64("paired-parent-order-v1", self.seed, epoch))
        return torch.randperm(len(self.dataset), generator=generator).tolist()

    def __iter__(self):
        epoch = int(self.dataset.epoch)
        if epoch < 0:
            raise RuntimeError("dataset epoch must be set before sampler iteration")
        return iter((index, epoch) for index in self.permutation(epoch))


class PairedSoftSaveDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, schedule_seed=42):
        self.reference = PairedSourceIndex(manifest_path, "reference")
        self.compressed = PairedSourceIndex(manifest_path, "compressed")
        self.schedule_seed = int(schedule_seed)
        self.epoch = -1
        self._shared_epoch = torch.tensor([-1], dtype=torch.int64).share_memory_()
        self.classes = self.reference.classes

    def __len__(self): return len(self.reference.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch); self._shared_epoch.fill_(int(epoch))

    def __getitem__(self, key):
        parent, key_epoch = map(int, key)
        shared = int(self._shared_epoch.item())
        if key_epoch != shared:
            raise RuntimeError(f"prefetch epoch mismatch key={key_epoch} shared={shared}")
        source = self.reference.source_index(parent, key_epoch)
        ref = self.reference.load(parent, source)
        comp = self.compressed.load(parent, source)
        flip = stable_u64("paired-soft-flip-v1", self.schedule_seed, key_epoch, parent) % 2 == 0
        if flip:
            ref = TF.hflip(ref); comp = TF.hflip(comp)
        ref = TF.normalize(TF.pil_to_tensor(ref).float().div_(255), AIRCRAFT_MEAN, AIRCRAFT_STD)
        comp = TF.normalize(TF.pil_to_tensor(comp).float().div_(255), AIRCRAFT_MEAN, AIRCRAFT_STD)
        coords = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32)
        return ref, comp, self.reference.targets[parent], bool(flip), coords, source, parent


class PairedSoftLoadDataset(torch.utils.data.Dataset):
    mode = "fkd_load"

    def __init__(self, manifest_path, image_mode, fkd_path, epochs, batch_size, sampler_seed=42):
        self.index = PairedSourceIndex(manifest_path, image_mode)
        self.classes = self.index.classes; self.targets = self.index.targets
        self.fkd_path = Path(fkd_path); self.args_epoch = int(epochs); self.args_bs = int(batch_size)
        self.epoch = -1; self._shared_epoch = torch.tensor([-1], dtype=torch.int64).share_memory_()
        self.sampler = PairedEpochSampler(self, sampler_seed)
        self.batch_config = None; self.batch_config_idx = 0
        if not (self.fkd_path / "relabel_manifest.json").is_file(): raise FileNotFoundError(self.fkd_path)

    def __len__(self): return len(self.index.rows)

    def set_epoch(self, epoch): self.epoch = int(epoch); self._shared_epoch.fill_(int(epoch))

    def load_batch_config(self, first_key):
        parent, epoch = map(int, first_key)
        if epoch != int(self._shared_epoch.item()): raise RuntimeError("load config epoch mismatch")
        order = self.sampler.permutation(epoch); position = order.index(parent)
        if position % self.args_bs: raise RuntimeError("first parent is not at batch boundary")
        batch = position // self.args_bs
        config = torch.load(self.fkd_path / f"epoch_{epoch}/batch_{batch}.tar", map_location="cpu", weights_only=False)
        if len(config) != 8: raise RuntimeError(f"paired FKD payload length {len(config)} != 8")
        self.batch_config_idx = 0
        self.batch_config = [config[0], config[1], config[6], config[7]]
        return config[2:6]

    def __getitem__(self, key):
        parent, epoch = map(int, key); position = self.batch_config_idx
        coords, flips, sources, parents = self.batch_config
        recorded_parent = int(parents[position]); source = int(sources[position])
        if parent != recorded_parent: raise RuntimeError(f"parent replay mismatch {parent}!={recorded_parent}")
        image = self.index.load(parent, source)
        if bool(flips[position]): image = TF.hflip(image)
        image = TF.normalize(TF.pil_to_tensor(image).float().div_(255), AIRCRAFT_MEAN, AIRCRAFT_STD)
        self.batch_config_idx += 1
        return image, self.targets[parent], bool(flips[position]), coords[position]


class PairedHardDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, image_mode, student_seed):
        self.index = PairedSourceIndex(manifest_path, image_mode)
        self.classes = self.index.classes; self.targets = self.index.targets
        self.student_seed = int(student_seed); self.epoch = -1
        self._shared_epoch = torch.tensor([-1], dtype=torch.int64).share_memory_()

    def __len__(self): return len(self.index.rows)

    def set_epoch(self, epoch): self.epoch = int(epoch); self._shared_epoch.fill_(int(epoch))

    def __getitem__(self, key):
        parent, key_epoch = map(int, key)
        if key_epoch != int(self._shared_epoch.item()): raise RuntimeError("hard prefetch epoch mismatch")
        source = self.index.source_index(parent, key_epoch)
        image = self.index.load(parent, source).resize((256, 256), Image.Resampling.BILINEAR)
        value = stable_u64("paired-hard-mild-rc-v1", self.student_seed, key_epoch, parent)
        top, left = value % 33, (value // 33) % 33
        flip = (value // (33 * 33)) % 2 == 0
        image = image.crop((left, top, left + 224, top + 224))
        if flip: image = TF.hflip(image)
        tensor = TF.normalize(TF.pil_to_tensor(image).float().div_(255), IMAGENET_MEAN, IMAGENET_STD)
        return tensor, self.targets[parent]

    def trajectory_audit(self, epochs):
        counts = np.zeros((len(self), 2), dtype=np.int64); digest = hashlib.sha256()
        sampler = PairedEpochSampler(self, self.student_seed)
        for epoch in range(epochs):
            for parent in sampler.permutation(epoch):
                source = self.index.source_index(parent, epoch); counts[parent, source] += 1
                value = stable_u64("paired-hard-mild-rc-v1", self.student_seed, epoch, parent)
                digest.update(f"{epoch}:{parent}:{source}:{value}\n".encode())
        return {"epochs": epochs, "examples": int(counts.sum()), "source_exposure_min": int(counts.min()),
                "source_exposure_max": int(counts.max()), "trajectory_sha256": digest.hexdigest()}
