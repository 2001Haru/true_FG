"""Compare RDED RRC-off/on reading views with CMMD and Teacher-feature Vendi."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from audit_cmmd_vendi_f1_k import (extract, file_sha256, load_teacher,
                                   unbiased_cmmd, vendi_summary)


class FKDViewDataset(Dataset):
    def __init__(self, image_root, fkd_root, epochs):
        image_root, fkd_root = Path(image_root), Path(fkd_root)
        samples = []
        for label, directory in enumerate(sorted(path for path in image_root.iterdir() if path.is_dir())):
            samples.extend((path, label) for path in sorted(path for path in directory.iterdir() if path.is_file()))
        if len(samples) != 300: raise RuntimeError((image_root, len(samples)))
        generator = torch.Generator().manual_seed(42); self.rows = []
        for epoch in range(400):
            order = torch.randperm(300, generator=generator)
            if epoch not in epochs: continue
            for batch in range(15):
                ids = order[batch * 20:(batch + 1) * 20]
                payload = torch.load(fkd_root / f"epoch_{epoch}/batch_{batch}.tar",
                                     map_location="cpu", weights_only=False)
                coords, flips = payload[:2]
                for position, index in enumerate(ids):
                    path, label = samples[int(index)]
                    self.rows.append((path, label, torch.as_tensor(coords[position]).float(), bool(flips[position])))
        if len(self.rows) != len(epochs) * 300: raise RuntimeError(len(self.rows))

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        path, label, coord, flip = self.rows[index]
        image = TF.pil_to_tensor(Image.open(path).convert("RGB"))
        if not torch.allclose(coord, torch.tensor([0., 0., 1., 1.]), rtol=0, atol=1e-7):
            top, left = float(coord[0]) * 224, float(coord[1]) * 224
            height, width = float(coord[2]) * 224, float(coord[3]) * 224
            image = TF.resized_crop(image, top, left, height, width, (224, 224),
                                    interpolation=InterpolationMode.BILINEAR, antialias=False)
        if flip: image = TF.hflip(image)
        return image, label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", action="append", required=True,
                        help="name=image root,FKD root")
    parser.add_argument("--reference-features", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--clip-source-root", required=True, type=Path)
    parser.add_argument("--clip-download-root", type=Path, default=Path("/root/.cache/clip"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--views", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    specs = {}
    for value in args.spec:
        name, paths = value.split("=", 1); image_root, fkd_root = paths.split(",", 1)
        specs[name] = (Path(image_root), Path(fkd_root))
    epochs = set(np.linspace(0, 399, args.views).round().astype(int).tolist())
    sys.path.insert(0, str(args.clip_source_root.resolve())); import clip
    clip_model, _ = clip.load("ViT-L/14@336px", device="cuda", jit=False,
                              download_root=str(args.clip_download_root)); clip_model = clip_model.float().eval()
    teacher = load_teacher(args.teacher); args.output_root.mkdir(parents=True, exist_ok=True)
    references = {}
    for name in ("test", "train", "r0"):
        payload = np.load(args.reference_features / f"{name}.npz")
        references[name] = torch.from_numpy(payload["clip"])
    groups = {}
    for name, (image_root, fkd_root) in specs.items():
        feature, teacher_feature, labels = extract(
            FKDViewDataset(image_root, fkd_root, epochs), clip_model, teacher,
            args.batch_size, args.workers, True, args.output_root / "features" / f"{name}.npz")
        groups[name] = {"cmmd_unbiased": {reference: unbiased_cmmd(feature, value)
                                           for reference, value in references.items()},
                        "vendi_teacher_cosine": vendi_summary(teacher_feature, labels)}
        print(name, json.dumps(groups[name]), flush=True)
    aggregate = {}
    for mode in ("off", "on"):
        names = sorted(name for name in groups if name.endswith("_" + mode))
        if not names: continue
        if len(names) != 3: raise RuntimeError((mode, names))
        aggregate[mode] = {"groups": names, "cmmd_unbiased": {}, "vendi_teacher_cosine": {}}
        for reference in references:
            values = [groups[name]["cmmd_unbiased"][reference] for name in names]
            aggregate[mode]["cmmd_unbiased"][reference] = {
                "mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)), "values": values}
        for metric in ("overall", "overall_fraction", "class_mean", "class_normalized_mean"):
            values = [groups[name]["vendi_teacher_cosine"][metric] for name in names]
            aggregate[mode]["vendi_teacher_cosine"][metric] = {
                "mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)), "values": values}
    output = {"status": "complete", "protocol": "aircraft_rded_rrc_cmmd_vendi_v1",
              "generation_seeds": [42, 43, 44], "epochs": sorted(epochs),
              "views_per_group": len(epochs) * 300,
              "view_definition": "replay exact FKD RRC coordinates and flip; exclude batch-level CutMix",
              "cmmd": {"model": "OpenAI CLIP ViT-L/14@336px", "sigma": 10., "scale": 1000.,
                       "estimator": "unbiased U-statistic"},
              "vendi": {"feature": "Teacher42 eval ResNet18 avgpool L2-normalized",
                        "kernel": "cosine", "primary": "class_mean"},
              "groups": groups, "aggregate": aggregate, "teacher_sha256": file_sha256(args.teacher)}
    (args.output_root / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__": main()
