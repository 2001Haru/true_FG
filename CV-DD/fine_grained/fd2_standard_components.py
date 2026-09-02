"""Auditable FD2 components for the standard fine-grained protocol.

This module intentionally keeps the two supported interpretations separate:
``released_semantics`` reproduces the behavior of the released FD2 scripts,
while ``paper_literal`` implements the equations in the ECCV 2026 paper.
It does not contain experiment orchestration and does not modify SRe2L++.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SEMANTICS = ("released_semantics", "paper_literal")


@dataclass(frozen=True)
class FD2DatasetSetting:
    attention_maps: int
    cal_ratio: float


FD2_DATASET_SETTINGS = {
    "CUB_imsize224": FD2DatasetSetting(attention_maps=8, cal_ratio=0.5),
    "A_imsize224": FD2DatasetSetting(attention_maps=32, cal_ratio=0.3),
    "SC_imsize224": FD2DatasetSetting(attention_maps=8, cal_ratio=0.3),
}


@dataclass(frozen=True)
class FD2SemanticSetting:
    name: str
    prototype_update: str
    teacher_cal_objective: str
    teacher_attention_crop_drop: bool
    recovery_class_objective: str
    recovery_cal_bn: bool
    previous_attention_view: str
    feature_weight: float
    similarity_weight: float
    group_size: int = 4
    intra_feature_weight: float = 0.5
    prototype_momentum: float = 0.05

    def to_dict(self) -> dict:
        return asdict(self)


SETTINGS = {
    "released_semantics": FD2SemanticSetting(
        name="released_semantics",
        prototype_update="released duplicate-label overwrite/count division",
        teacher_cal_objective="released crop/drop auxiliary CAL objective",
        teacher_attention_crop_drop=True,
        recovery_class_objective="replace baseline CE with (1-alpha)*CE_backbone+alpha*CE_CAL",
        recovery_cal_bn=True,
        previous_attention_view="one cached RandomResizedCrop+HorizontalFlip view",
        feature_weight=0.9,
        similarity_weight=0.1,
    ),
    "paper_literal": FD2SemanticSetting(
        name="paper_literal",
        prototype_update="Eq.(5), classwise mean for duplicate labels",
        teacher_cal_objective="Eq.(6): CE_raw+CE_effect+eta*center (eta=1)",
        teacher_attention_crop_drop=False,
        recovery_class_objective="retain baseline CE+BN and add Eq.(11) L_cls",
        recovery_cal_bn=False,
        previous_attention_view="direct saved-image view from Algorithm 1",
        feature_weight=0.8,
        similarity_weight=0.2,
    ),
}


def semantic_setting(name: str) -> FD2SemanticSetting:
    try:
        return SETTINGS[name]
    except KeyError as error:
        raise ValueError(f"Unknown FD2 semantics {name!r}; choose one of {SEMANTICS}") from error


def dataset_setting(name: str) -> FD2DatasetSetting:
    try:
        return FD2_DATASET_SETTINGS[name]
    except KeyError as error:
        raise ValueError(f"No FD2 setting for dataset {name!r}") from error


def group_start(ipc_id: int, group_size: int = 4) -> int:
    if ipc_id < 0:
        raise ValueError("ipc_id must be non-negative")
    return (ipc_id // group_size) * group_size


def previous_ids_in_group(ipc_id: int, group_size: int = 4) -> tuple[int, ...]:
    return tuple(range(group_start(ipc_id, group_size), ipc_id))


class LastFeatureHook:
    def __init__(self, module: nn.Module):
        self.feature = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        self.feature = output

    def close(self) -> None:
        self.handle.remove()


class BNFeatureHook:
    def __init__(self, module: nn.BatchNorm2d):
        self.r_feature = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, _output):
        value = inputs[0]
        channels = value.shape[1]
        mean = value.mean((0, 2, 3))
        variance = value.permute(1, 0, 2, 3).contiguous().reshape(channels, -1).var(1, unbiased=False)
        self.r_feature = torch.norm(module.running_var.detach() - variance, 2) + torch.norm(
            module.running_mean.detach() - mean, 2
        )

    def close(self) -> None:
        self.handle.remove()


class BasicConv2d(nn.Module):
    """The released CAL 1x1 attention block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(inputs)), inplace=True)


class BAP(nn.Module):
    def forward(self, features: torch.Tensor, attentions: torch.Tensor):
        batch, _channels, height, width = features.shape
        if attentions.shape[-2:] != (height, width):
            attentions = F.interpolate(attentions, size=(height, width), mode="bilinear", align_corners=False)
        feature_matrix = torch.einsum("imjk,injk->imn", attentions, features)
        feature_matrix = feature_matrix.div(float(height * width)).reshape(batch, -1)
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + 1e-6)
        feature_matrix = F.normalize(feature_matrix, dim=-1)
        fake_attention = torch.empty_like(attentions).uniform_(0, 2) if self.training else torch.ones_like(attentions)
        counterfactual = torch.einsum("imjk,injk->imn", fake_attention, features)
        counterfactual = counterfactual.div(float(height * width)).reshape(batch, -1)
        counterfactual = torch.sign(counterfactual) * torch.sqrt(torch.abs(counterfactual) + 1e-6)
        return feature_matrix, F.normalize(counterfactual, dim=-1)


class CAL(nn.Module):
    """ResNet18 CAL copied semantically from the released FD2 implementation."""

    def __init__(self, num_classes: int, attention_maps: int):
        super().__init__()
        self.M = attention_maps
        self.num_features = 512
        self.attentions = BasicConv2d(self.num_features, attention_maps)
        self.bap = BAP()
        self.fc = nn.Linear(attention_maps * self.num_features, num_classes, bias=False)

    def forward(self, feature_maps: torch.Tensor):
        batch_size = feature_maps.shape[0]
        attention_maps = self.attentions(feature_maps)
        feature_matrix, counterfactual = self.bap(feature_maps, attention_maps)
        raw = self.fc(feature_matrix * 100.0)
        effect = raw - self.fc(counterfactual * 100.0)
        if self.training:
            selected = []
            for sample in range(batch_size):
                weights = torch.sqrt(attention_maps[sample].sum(dim=(1, 2)).detach() + 1e-6)
                probabilities = F.normalize(weights, p=1, dim=0).cpu().numpy()
                indices = np.random.choice(self.M, 2, p=probabilities)
                selected.append(attention_maps[sample, indices])
            attention = torch.stack(selected)
        else:
            attention = attention_maps.mean(dim=1, keepdim=True)
        return raw, effect, feature_matrix, attention, attention_maps


class CenterLoss(nn.Module):
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(outputs, targets, reduction="sum") / outputs.size(0)


def batch_augment(
    images: torch.Tensor,
    attention_map: torch.Tensor,
    mode: str,
    theta,
    padding_ratio: float = 0.1,
) -> torch.Tensor:
    batch, _channels, height, width = images.shape
    if mode == "crop":
        crops = []
        for index in range(batch):
            current = attention_map[index : index + 1]
            threshold = (random.uniform(*theta) if isinstance(theta, tuple) else theta) * current.max()
            mask = F.interpolate(current, size=(height, width), mode="bilinear", align_corners=False) >= threshold
            nonzero = torch.nonzero(mask[0, 0])
            y0 = max(int(nonzero[:, 0].min() - padding_ratio * height), 0)
            y1 = min(int(nonzero[:, 0].max() + padding_ratio * height), height)
            x0 = max(int(nonzero[:, 1].min() - padding_ratio * width), 0)
            x1 = min(int(nonzero[:, 1].max() + padding_ratio * width), width)
            crops.append(F.interpolate(images[index : index + 1, :, y0:y1, x0:x1], size=(height, width), mode="bilinear", align_corners=False))
        return torch.cat(crops)
    if mode == "drop":
        masks = []
        for index in range(batch):
            current = attention_map[index : index + 1]
            threshold = (random.uniform(*theta) if isinstance(theta, tuple) else theta) * current.max()
            masks.append(F.interpolate(current, size=(height, width), mode="bilinear", align_corners=False) < threshold)
        return images * torch.cat(masks).float()
    raise ValueError(f"Unsupported batch augmentation {mode!r}")


@torch.no_grad()
def update_feature_centers(
    centers: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    semantics: str,
    momentum: float = 0.05,
) -> None:
    """Update centers in place, preserving each interpretation's duplicate-label behavior."""
    if semantics == "released_semantics":
        normalized_centers = F.normalize(centers[labels], dim=-1)
        delta = torch.zeros_like(centers)
        # Advanced-index assignment deliberately preserves the released last-write behavior.
        delta[labels] = features.detach() - normalized_centers
        counts = torch.zeros(centers.shape[0], device=centers.device, dtype=centers.dtype)
        counts.scatter_add_(0, labels, torch.ones_like(labels, dtype=centers.dtype))
        centers.add_(momentum * delta / counts.clamp_min(1).unsqueeze(1))
        return
    if semantics == "paper_literal":
        detached = F.normalize(features.detach(), dim=-1)
        for label in labels.unique(sorted=True):
            class_id = int(label.item())
            class_feature = detached[labels == label].mean(dim=0)
            class_feature = F.normalize(class_feature, dim=0)
            centers[class_id].mul_(1.0 - momentum).add_(class_feature, alpha=momentum)
        return
    semantic_setting(semantics)


def normalized_l2(left: torch.Tensor, right: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    numerator = torch.norm(left - right, p=2, dim=dim)
    denominator = torch.norm(left, p=2, dim=dim) + torch.norm(right, p=2, dim=dim) + eps
    return numerator / denominator


def fine_grained_characteristic_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    centers: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    intra = normalized_l2(features, centers[labels]).mean()
    # Algebraically identical to pairwise torch.norm(feature-center), without
    # materializing [batch, classes, M*channels] (over 1 GiB for Aircraft).
    feature_norm = torch.norm(features, p=2, dim=1)
    center_norm = torch.norm(centers, p=2, dim=1)
    squared = (
        feature_norm.square().unsqueeze(1)
        + center_norm.square().unsqueeze(0)
        - 2.0 * features @ centers.t()
    ).clamp_min(0.0)
    distances = torch.sqrt(squared) / (
        feature_norm.unsqueeze(1) + center_norm.unsqueeze(0) + 1e-6
    )
    own_class = F.one_hot(labels, num_classes=centers.shape[0]).bool()
    inter = distances.masked_fill(own_class, 0.0).sum(dim=1).div(centers.shape[0] - 1).mean()
    return beta * intra + (1.0 - beta) * (1.0 - inter)


def similarity_loss(current_attention: torch.Tensor, previous_attention: torch.Tensor) -> torch.Tensor:
    """Eq.(10): minimize one minus the normalized L2 attention distance."""
    current = current_attention.reshape(current_attention.shape[0], -1)
    previous = previous_attention.reshape(previous_attention.shape[0], -1)
    distances = normalized_l2(previous, current.expand_as(previous), dim=1)
    return 1.0 - distances.mean()


def compose_recovery_loss(
    semantics: str,
    *,
    ce_backbone: torch.Tensor,
    ce_cal: torch.Tensor,
    bn_backbone: torch.Tensor,
    bn_cal: torch.Tensor,
    feature_loss: torch.Tensor,
    similarity: torch.Tensor,
    cal_ratio: float,
    r_bn: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    setting = semantic_setting(semantics)
    class_loss = (1.0 - cal_ratio) * ce_backbone + cal_ratio * ce_cal
    if semantics == "released_semantics":
        other = r_bn * (bn_backbone + bn_cal)
        total = class_loss + other
    else:
        # Literal Eq.(12): original SRe2L++ objective plus the new Eq.(11) term.
        other = ce_backbone + r_bn * bn_backbone
        total = other + class_loss
    total = total + setting.feature_weight * feature_loss + setting.similarity_weight * similarity
    return total, {
        "other": other,
        "class": class_loss,
        "feature": feature_loss,
        "similarity": similarity,
    }
