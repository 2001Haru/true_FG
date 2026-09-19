"""CMMD under one frozen view schedule for R0, RDED, and packed F1 images.

The main estimate uses 32 views per parent (N=9600).  The N=600
sensitivity estimate is a strict nested subset: the first two selected
epochs for every parent.  CutMix is deliberately absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from audit_cmmd_vendi_f1_k import extract, unbiased_cmmd
from paired_source_dataset import PairedSourceIndex


class CommonSchedule:
    """Read one canonical RRC trajectory and expose it by (epoch,parent)."""

    def __init__(self, fkd_root: Path, epochs: list[int]):
        self.rows: dict[tuple[int, int], tuple[torch.Tensor, bool]] = {}
        generator = torch.Generator().manual_seed(42)
        digest = hashlib.sha256()
        selected = set(epochs)
        for epoch in range(400):
            order = torch.randperm(300, generator=generator)
            if epoch not in selected:
                continue
            seen: list[int] = []
            for batch in range(15):
                payload = torch.load(fkd_root / f"epoch_{epoch}/batch_{batch}.tar",
                                     map_location="cpu", weights_only=False)
                coords, flips = payload[:2]
                if len(coords) != 20 or len(flips) != 20:
                    raise RuntimeError((epoch, batch, len(coords), len(flips)))
                for position, parent_tensor in enumerate(order[batch * 20:(batch + 1) * 20]):
                    parent = int(parent_tensor)
                    coord = torch.as_tensor(coords[position]).float().clone()
                    flip = bool(flips[position])
                    self.rows[(epoch, parent)] = (coord, flip)
                    digest.update(f"{epoch}:{parent}:{coord.tolist()}:{int(flip)}\n".encode())
                    seen.append(parent)
            if sorted(seen) != list(range(300)):
                raise RuntimeError(f"epoch {epoch} is not a permutation")
        if len(self.rows) != len(epochs) * 300:
            raise RuntimeError((len(self.rows), len(epochs) * 300))
        self.sha256 = digest.hexdigest()


class CommonViewDataset(Dataset):
    def __init__(self, schedule: CommonSchedule, epochs: list[int], policy: str,
                 image_root: Path | None = None, paired_manifest: Path | None = None):
        if (image_root is None) == (paired_manifest is None):
            raise ValueError("exactly one of image_root/paired_manifest is required")
        if policy not in ("off", "on"):
            raise ValueError(policy)
        self.schedule, self.epochs, self.policy = schedule, epochs, policy
        self.index = PairedSourceIndex(paired_manifest, "compressed_fir") if paired_manifest else None
        if self.index is not None:
            if len(self.index.rows) != 300:
                raise RuntimeError((paired_manifest, len(self.index.rows)))
            self.targets = self.index.targets
        else:
            samples = []
            for label, directory in enumerate(sorted(path for path in image_root.iterdir() if path.is_dir())):
                samples.extend((path, label) for path in sorted(path for path in directory.iterdir() if path.is_file()))
            if len(samples) != 300:
                raise RuntimeError((image_root, len(samples)))
            self.samples = samples
        # Epoch-major and then parent-major supports a common stratified
        # two-view-per-parent subset of the full 32-view set.
        self.rows = [(epoch, parent) for epoch in epochs for parent in range(300)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        epoch, parent = self.rows[item]
        if self.index is None:
            path, label = self.samples[parent]
            image = Image.open(path).convert("RGB")
        else:
            source = self.index.source_index(parent, epoch)
            image = self.index.load(parent, source)
            label = self.targets[parent]
        # Match the audited FKD replay path: convert the stored 8-bit image to
        # a tensor before cropping, and use bilinear without antialiasing.
        image = TF.pil_to_tensor(image)
        coord, flip = self.schedule.rows[(epoch, parent)]
        if self.policy == "on":
            top, left = round(float(coord[0]) * image.shape[-2]), round(float(coord[1]) * image.shape[-1])
            height, width = round(float(coord[2]) * image.shape[-2]), round(float(coord[3]) * image.shape[-1])
            if min(height, width) <= 0 or top + height > image.shape[-2] or left + width > image.shape[-1]:
                raise RuntimeError((epoch, parent, coord.tolist(), tuple(image.shape)))
            image = TF.resized_crop(image, top, left, height, width, (224, 224),
                                    interpolation=InterpolationMode.BILINEAR, antialias=False)
        elif tuple(image.shape[-2:]) != (224, 224):
            image = TF.resize(image, (224, 224), interpolation=InterpolationMode.BILINEAR,
                              antialias=False)
        if flip:
            image = TF.hflip(image)
        return image, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-spec", action="append", default=[], help="name=image root")
    parser.add_argument("--paired-spec", action="append", default=[], help="name=paired manifest")
    parser.add_argument("--policy", choices=("off", "on"), required=True)
    parser.add_argument("--geometry-fkd", required=True, type=Path)
    parser.add_argument("--reference-features", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--clip-source-root", required=True, type=Path)
    parser.add_argument("--clip-download-root", type=Path, default=Path("/root/.cache/clip"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--views", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    epochs = sorted(set(np.linspace(0, 399, args.views).round().astype(int).tolist()))
    if len(epochs) != args.views:
        raise RuntimeError((args.views, epochs))
    schedule = CommonSchedule(args.geometry_fkd, epochs)
    specs = []
    for item in args.image_spec:
        name, path = item.split("=", 1); specs.append((name, Path(path), None))
    for item in args.paired_spec:
        name, path = item.split("=", 1); specs.append((name, None, Path(path)))
    if len({name for name, _, _ in specs}) != len(specs):
        raise RuntimeError("duplicate group name")
    sys.path.insert(0, str(args.clip_source_root.resolve()))
    import clip
    clip_model, _ = clip.load("ViT-L/14@336px", device="cuda", jit=False,
                              download_root=str(args.clip_download_root))
    clip_model = clip_model.float().eval()
    references = {name: torch.from_numpy(np.load(args.reference_features / f"{name}.npz")["clip"])
                  for name in ("test", "train", "r0")}
    # Frozen before looking at any metric: two of the 32 view positions for
    # every parent.  Per-parent sampling avoids over-weighting any source slot
    # while preserving exact class balance and nesting in N=9600.
    sensitivity_indices = []
    for parent in range(300):
        chosen = np.random.default_rng(20260919 + parent).choice(len(epochs), size=2, replace=False)
        sensitivity_indices.extend(int(position) * 300 + parent for position in sorted(chosen))
    sensitivity_indices = torch.tensor(sorted(sensitivity_indices), dtype=torch.long)
    sensitivity_digest = hashlib.sha256(sensitivity_indices.numpy().tobytes()).hexdigest()
    args.output_root.mkdir(parents=True, exist_ok=True)
    groups = {}
    for name, image_root, manifest in specs:
        dataset = CommonViewDataset(schedule, epochs, args.policy, image_root, manifest)
        feature, _, labels = extract(dataset, clip_model, None, args.batch_size, args.workers, False,
                                     args.output_root / "features" / f"{name}_{args.policy}.npz")
        if len(feature) != 9600 or len(labels) != 9600:
            raise RuntimeError((name, len(feature), len(labels)))
        if sorted(torch.bincount(labels, minlength=100).tolist()) != [96] * 100:
            raise RuntimeError(f"unbalanced labels for {name}")
        if sorted(torch.bincount(labels[sensitivity_indices], minlength=100).tolist()) != [6] * 100:
            raise RuntimeError(f"unbalanced N=600 subset for {name}")
        groups[name] = {}
        for count in (600, 9600):
            subset = feature[sensitivity_indices] if count == 600 else feature
            groups[name][f"n{count}"] = {reference: unbiased_cmmd(subset, value)
                                          for reference, value in references.items()}
        print(name, json.dumps(groups[name]), flush=True)
    result = {
        "status": "complete", "protocol": "aircraft_cmmd_common_views_v1",
        "policy": args.policy, "cutmix": False, "epochs": epochs,
        "main_n": 9600, "sensitivity_n": 600,
        "n600_definition": "two pre-registered view positions per parent selected with NumPy seed "
                           "20260919+parent; exact balanced subset of N=9600",
        "n600_indices_sha256": sensitivity_digest,
        "view_definition": "one canonical R0 seed0 RRC trajectory shared by every construction; "
                           "RRC-off replaces crop coordinates by identity but retains the same flips",
        "geometry_fkd": str(args.geometry_fkd.resolve()),
        "geometry_schedule_sha256": schedule.sha256,
        "cmmd": {"model": "OpenAI CLIP ViT-L/14@336px", "sigma": 10.0, "scale": 1000.0,
                 "estimator": "unbiased U-statistic Gaussian MMD"},
        "references": {name: len(value) for name, value in references.items()},
        "groups": groups,
    }
    (args.output_root / f"summary_{args.policy}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
