"""Zero-training audit of F1's dataset-BN target versus GN's sample-group target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision import models

from paired_source_dataset import AIRCRAFT_MEAN, AIRCRAFT_STD, PairedSourceIndex


def load_images(index, keys):
    rows = []
    for parent, source in keys:
        array = np.asarray(index.load(parent, source), dtype=np.uint8).copy()
        rows.append(torch.from_numpy(array).permute(2, 0, 1).float().div_(255))
    return torch.stack(rows)


class Accumulator:
    def __init__(self, channels=64, groups=32):
        self.sum = {arm: torch.zeros(channels, dtype=torch.float64) for arm in ("clear", "bbox", "f1")}
        self.sumsq = {arm: torch.zeros(channels, dtype=torch.float64) for arm in ("clear", "bbox", "f1")}
        self.count = 0; self.groups = groups
        self.group_loss = {arm: [0.0, 0.0, 0] for arm in ("bbox", "f1")}

    def add(self, values):
        for arm, activation in values.items():
            value = activation.detach().double().cpu()
            self.sum[arm] += value.sum((0, 2, 3)); self.sumsq[arm] += value.square().sum((0, 2, 3))
        self.count += values["clear"].shape[0] * values["clear"].shape[2] * values["clear"].shape[3]
        clear = values["clear"].reshape(values["clear"].shape[0], self.groups, -1)
        cm, cv = clear.mean(2), clear.var(2, unbiased=False)
        for arm in ("bbox", "f1"):
            candidate = values[arm].reshape(values[arm].shape[0], self.groups, -1)
            mean, var = candidate.mean(2), candidate.var(2, unbiased=False)
            mean_loss = ((mean - cm).square() / (cv + 1e-5)).mean(1)
            logvar_loss = ((var + 1e-5).log() - (cv + 1e-5).log()).square().mean(1)
            self.group_loss[arm][0] += float(mean_loss.sum())
            self.group_loss[arm][1] += float(logvar_loss.sum())
            self.group_loss[arm][2] += len(mean_loss)

    def finish(self):
        moments = {}
        for arm in self.sum:
            mean = self.sum[arm] / self.count
            var = self.sumsq[arm] / self.count - mean.square()
            moments[arm] = (mean, var.clamp_min(0))
        target_mean, target_var = moments["clear"]
        output = {}
        for arm in ("bbox", "f1"):
            mean, var = moments[arm]
            dataset_mean = ((mean - target_mean).square() / (target_var + 1e-6)).mean()
            dataset_logvar = ((var + 1e-6).log() - (target_var + 1e-6).log()).square().mean()
            gm, gv, count = self.group_loss[arm]
            output[arm] = {
                "dataset_channel_mean": float(dataset_mean),
                "dataset_channel_logvar": float(dataset_logvar),
                "dataset_total": float(dataset_mean + dataset_logvar),
                "sample_group_mean": gm / count,
                "sample_group_logvar": gv / count,
                "sample_group_total": (gm + gv) / count,
            }
        return output


@torch.no_grad()
def evaluate(conv, indices, batch_size, device):
    accumulator = Accumulator(channels=conv.out_channels, groups=32)
    mean = torch.tensor(AIRCRAFT_MEAN, device=device)[None, :, None, None]
    std = torch.tensor(AIRCRAFT_STD, device=device)[None, :, None, None]
    keys = [(parent, source) for parent in range(len(indices["clear"].rows))
            for source in range(indices["clear"].sources_per_parent)]
    for start in range(0, len(keys), batch_size):
        chunk = keys[start:start + batch_size]
        values = {arm: conv((load_images(index, chunk).to(device) - mean) / std)
                  for arm, index in indices.items()}
        accumulator.add(values)
    return accumulator.finish(), len(keys)


def reductions(metrics):
    result = {}
    for metric in ("dataset_channel_mean", "dataset_channel_logvar", "dataset_total",
                   "sample_group_mean", "sample_group_logvar", "sample_group_total"):
        result[metric] = 1.0 - metrics["f1"][metric] / metrics["bbox"][metric]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox-manifest", required=True, type=Path)
    parser.add_argument("--f1-manifest", required=True, type=Path)
    parser.add_argument("--r18-weights", required=True, type=Path)
    parser.add_argument("--gn-model", default="resnet50_gn.a1h_in1k")
    parser.add_argument("--gn-weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(); device = torch.device("cuda")
    clear = PairedSourceIndex(args.bbox_manifest, "reference")
    bbox = PairedSourceIndex(args.bbox_manifest, "compressed")
    f1 = PairedSourceIndex(args.f1_manifest, "compressed_fir")
    if clear.targets != bbox.targets or clear.targets != f1.targets or clear.sources_per_parent != 2:
        raise RuntimeError("paired source mismatch")
    indices = {"clear": clear, "bbox": bbox, "f1": f1}
    r18 = models.resnet18(weights=None)
    r18.load_state_dict(torch.load(args.r18_weights, map_location="cpu", weights_only=True))
    r18conv = r18.conv1.to(device).eval().requires_grad_(False)
    import timm
    gn = timm.create_model(args.gn_model, pretrained=False, num_classes=1000, drop_path_rate=0.0)
    gn.load_state_dict(torch.load(args.gn_weights, map_location="cpu", weights_only=True))
    if getattr(gn.bn1, "num_groups", None) != 32 or gn.conv1.out_channels != 64:
        raise RuntimeError((gn.bn1, gn.conv1))
    gnconv = gn.conv1.to(device).eval().requires_grad_(False)
    r18_metrics, images = evaluate(r18conv, indices, args.batch_size, device)
    gn_metrics, images2 = evaluate(gnconv, indices, args.batch_size, device)
    if images != images2 or images != 600: raise RuntimeError((images, images2))
    r18_reduction, gn_reduction = reductions(r18_metrics), reductions(gn_metrics)
    primary_dataset = r18_reduction["dataset_total"]
    primary_group = gn_reduction["sample_group_total"]
    output = {
        "status": "complete", "protocol": "aircraft_f1_gn_objective_orthogonality_v1",
        "images": images, "views": "stored full-frame decoded sources; no flip/RRC/CutMix",
        "normalization": {"mean": AIRCRAFT_MEAN, "std": AIRCRAFT_STD},
        "definitions": {
            "dataset": "per-channel moments over N,H,W; standardized mean MSE + log-variance MSE",
            "sample_group": "per-image moments over each of 32 groups' channels,H,W; same loss then average N,G",
        },
        "r18_f1_objective": {"metrics": r18_metrics, "reductions": r18_reduction},
        "gn50_actual_objective": {"metrics": gn_metrics, "reductions": gn_reduction},
        "preregistered_primary": {
            "dataset_reduction": primary_dataset, "sample_group_reduction": primary_group,
            "dataset_minus_group_reduction": primary_dataset - primary_group,
            "dataset_reduction_much_greater": bool(primary_dataset >= primary_group + .5),
            "group_reduction_near_zero": bool(abs(primary_group) <= .1),
            "gate_to_fit_f1_gn": bool(primary_dataset >= primary_group + .5),
            "thresholds_frozen_before_execution": "much greater := gap>=0.50; near zero := |reduction|<=0.10",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
