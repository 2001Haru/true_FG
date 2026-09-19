"""CMMD and Vendi diagnostics for Aircraft k=2/3/4 F1-v1 and k=4 F1-v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import functional as TF

from paired_source_dataset import PairedSourceIndex


CLIP_MEAN = (.48145466, .4578275, .40821073)
CLIP_STD = (.26862954, .26130258, .27577711)
AIRCRAFT_MEAN = (.4865, .5177, .5425)
AIRCRAFT_STD = (.2124, .2051, .2375)


class PathDataset(Dataset):
    def __init__(self, root):
        root = Path(root); self.rows = []
        for label, directory in enumerate(sorted(path for path in root.iterdir() if path.is_dir())):
            self.rows.extend((path, label) for path in sorted(path for path in directory.iterdir() if path.is_file()))

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        path, label = self.rows[index]
        image = Image.open(path).convert("RGB")
        return TF.pil_to_tensor(image), label


class PairedDecodedDataset(Dataset):
    def __init__(self, manifest):
        self.index = PairedSourceIndex(manifest, "compressed_fir")
        self.rows = [(parent, source, self.index.targets[parent])
                     for parent in range(len(self.index.rows))
                     for source in range(self.index.sources_per_parent)]

    def __len__(self): return len(self.rows)

    def __getitem__(self, item):
        parent, source, label = self.rows[item]
        return TF.pil_to_tensor(self.index.load(parent, source)), label


def load_teacher(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "model" in state: state = state["model"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model = models.resnet18(weights=None); model.fc = nn.Linear(model.fc.in_features, 100)
    model.load_state_dict(state)
    return model.cuda().eval()


def normalize(images, mean, std):
    mean = torch.tensor(mean, device=images.device)[None, :, None, None]
    std = torch.tensor(std, device=images.device)[None, :, None, None]
    return (images - mean) / std


@torch.no_grad()
def extract(dataset, clip_model, teacher, batch_size, workers, need_teacher, cache):
    if cache.is_file():
        payload = np.load(cache)
        return torch.from_numpy(payload["clip"]), (torch.from_numpy(payload["teacher"])
                if "teacher" in payload.files else None), torch.from_numpy(payload["labels"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        persistent_workers=workers > 0, pin_memory=True)
    clip_rows, teacher_rows, labels = [], [], []
    captured = {}
    handle = (teacher.avgpool.register_forward_hook(
        lambda _, __, output: captured.__setitem__("feature", output.flatten(1).detach()))
        if need_teacher else None)
    for images, target in loader:
        images = images.cuda(non_blocking=True).float().div_(255)
        clip_input = F.interpolate(images, size=(336, 336), mode="bicubic", align_corners=False, antialias=True)
        clip_input = normalize(clip_input, CLIP_MEAN, CLIP_STD)
        clip_feature = clip_model.encode_image(clip_input)
        clip_rows.append(F.normalize(clip_feature.float(), dim=1).cpu())
        if need_teacher:
            teacher(normalize(images, AIRCRAFT_MEAN, AIRCRAFT_STD))
            teacher_rows.append(F.normalize(captured["feature"].float(), dim=1).cpu())
        labels.append(target.cpu())
    if handle is not None: handle.remove()
    clip_rows = torch.cat(clip_rows); labels = torch.cat(labels)
    teacher_rows = torch.cat(teacher_rows) if need_teacher else None
    cache.parent.mkdir(parents=True, exist_ok=True)
    values = {"clip": clip_rows.numpy(), "labels": labels.numpy()}
    if teacher_rows is not None: values["teacher"] = teacher_rows.numpy()
    np.savez(cache, **values)
    return clip_rows, teacher_rows, labels


def kernel_sum(left, right, sigma=10., chunk=512):
    left = left.cuda().double(); right = right.cuda().double(); total = 0.
    gamma = 1. / (2 * sigma * sigma)
    for start in range(0, len(left), chunk):
        value = left[start:start + chunk]
        distance = (value.square().sum(1)[:, None] + right.square().sum(1)[None, :] -
                    2 * value @ right.T).clamp_min_(0)
        total += float(torch.exp(-gamma * distance).sum().cpu())
    return total


def unbiased_cmmd(left, right, sigma=10., scale=1000.):
    n, m = len(left), len(right)
    xx = (kernel_sum(left, left, sigma) - n) / (n * (n - 1))
    yy = (kernel_sum(right, right, sigma) - m) / (m * (m - 1))
    xy = kernel_sum(left, right, sigma) / (n * m)
    return scale * (xx + yy - 2 * xy)


def vendi(features):
    features = F.normalize(features.cuda().double(), dim=1)
    # K/N and Z^T Z/N have identical non-zero eigenvalues.  Use the smaller
    # side so view-level audits with thousands of samples remain exact.
    kernel = (features @ features.T if len(features) <= features.shape[1]
              else features.T @ features)
    eigenvalues = torch.linalg.eigvalsh(kernel).clamp_min_(0)
    eigenvalues = eigenvalues / eigenvalues.sum()
    positive = eigenvalues[eigenvalues > 1e-15]
    return float(torch.exp(-(positive * positive.log()).sum()))


def vendi_summary(features, labels):
    overall = vendi(features)
    classes = sorted(map(int, labels.unique()))
    per_class = [vendi(features[labels == label]) for label in classes]
    per_class_sizes = [int((labels == label).sum()) for label in classes]
    return {"overall": overall, "overall_fraction": overall / len(features),
            "class_mean": float(np.mean(per_class)), "class_sd": float(np.std(per_class, ddof=1)),
            "class_min": float(np.min(per_class)), "class_max": float(np.max(per_class)),
            "class_normalized_mean": float(np.mean(np.asarray(per_class) / np.asarray(per_class_sizes))),
            "samples": len(features), "classes": len(classes),
            "samples_per_class_min": min(per_class_sizes), "samples_per_class_max": max(per_class_sizes)}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", action="append", required=True,
                        help="name=paired manifest")
    parser.add_argument("--r0-root", required=True, type=Path)
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--test-root", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--clip-source-root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "third_party/metric_backbones/CLIP")
    parser.add_argument("--clip-download-root", type=Path, default=Path("/root/.cache/clip"))
    args = parser.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    specs = dict(value.split("=", 1) for value in args.spec)
    args.output_root.mkdir(parents=True, exist_ok=True); cache = args.output_root / "features"
    sys.path.insert(0, str(args.clip_source_root.resolve()))
    import clip
    clip_model, _ = clip.load("ViT-L/14@336px", device="cuda", jit=False,
                              download_root=str(args.clip_download_root))
    clip_model = clip_model.float().eval()
    clip_weight = args.clip_download_root / "ViT-L-14-336px.pt"
    if not clip_weight.is_file(): raise FileNotFoundError(clip_weight)
    teacher = load_teacher(args.teacher)
    references = {}
    for name, root in (("test", args.test_root), ("train", args.train_root), ("r0", args.r0_root)):
        clip_feature, _, labels = extract(PathDataset(root), clip_model, teacher, args.batch_size,
                                          args.workers, False, cache / f"{name}.npz")
        references[name] = {"clip": clip_feature, "labels": labels}
    groups = {}
    for name, manifest in specs.items():
        clip_feature, teacher_feature, labels = extract(
            PairedDecodedDataset(manifest), clip_model, teacher, args.batch_size, args.workers, True,
            cache / f"{name}.npz")
        groups[name] = {"cmmd_unbiased": {
            reference: unbiased_cmmd(clip_feature, payload["clip"])
            for reference, payload in references.items()},
            "vendi_teacher_cosine": vendi_summary(teacher_feature, labels)}
        print(name, json.dumps(groups[name]), flush=True)
    # Freeze the two known performance readings on the same image groups.
    performance = {
        "k2_f1_v1": {"raw_final": 75.8776, "public_bn_ceiling": 76.2276},
        "k3_f1_v1": {"raw_final": 76.6877, "public_bn_ceiling": 77.4778},
        "k4_f1_v1": {"raw_final": 76.4276, "public_bn_ceiling": 78.1278},
        "k4_f1_v2": {"raw_final": 75.8376, "public_bn_ceiling": 78.1278},
    }
    output = {"status": "complete", "protocol": "aircraft_f1_k_cmmd_vendi_v1",
              "interpretation": "raw and ceiling are two Student readings for the same decoded image sets, not separate image conditions",
              "clip": {"model": "OpenAI CLIP ViT-L/14@336px", "source_revision": "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6",
                       "weight_sha256": file_sha256(clip_weight),
                       "input": "RGB[0,1] -> bicubic antialiased warp336 -> OpenAI CLIP normalization",
                       "embedding": "projected 768D image embedding, L2 normalized as in official Scenic CLIP default"},
              "cmmd": {"estimator": "unbiased U-statistic Gaussian MMD", "sigma": 10., "scale": 1000.,
                       "deviation_from_official_code": "user-requested unbiased estimate; current official code uses minimum-variance biased estimate"},
              "vendi": {"feature": "Aircraft Teacher42 eval-mode ResNet18 avgpool 512D, L2 normalized",
                        "kernel": "cosine Gram matrix; eigenvalues of K/trace(K); exp Shannon entropy"},
              "references": {name: {"samples": len(value["clip"])} for name, value in references.items()},
              "performance": performance, "groups": groups,
              "teacher_sha256": file_sha256(args.teacher)}
    output_path = args.output_root / "summary.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
