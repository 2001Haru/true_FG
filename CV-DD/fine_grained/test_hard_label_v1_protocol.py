"""Unit tests for the frozen hard-label v1 scientific invariants."""

import math
import json
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms


HERE = Path(__file__).resolve().parent
VALIDATE_DIR = HERE.parent / "validate"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(VALIDATE_DIR))

from hard_label_v1_protocol import (  # noqa: E402
    BACKBONE_LR,
    HEAD_LR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMAGENET_V1_FILE_SHA256,
    TOTAL_UPDATES,
    cosine_lr,
)
from train_hard_label_v1 import make_transforms, split_parameter_groups  # noqa: E402


class HardLabelV1ProtocolTest(unittest.TestCase):
    def test_cosine_endpoints_midpoint_and_group_ratio(self):
        self.assertEqual(cosine_lr(BACKBONE_LR, 0.0, 0, TOTAL_UPDATES), BACKBONE_LR)
        self.assertEqual(cosine_lr(BACKBONE_LR, 0.0, TOTAL_UPDATES, TOTAL_UPDATES), 0.0)
        self.assertTrue(
            math.isclose(
                cosine_lr(BACKBONE_LR, 0.0, TOTAL_UPDATES // 2, TOTAL_UPDATES),
                BACKBONE_LR / 2,
                abs_tol=1e-15,
            )
        )
        for t in (0, 1, 299, 1500, 2999):
            backbone = cosine_lr(BACKBONE_LR, 0.0, t, TOTAL_UPDATES)
            head = cosine_lr(HEAD_LR, 0.0, t, TOTAL_UPDATES)
            self.assertTrue(math.isclose(head / backbone, 10.0, abs_tol=1e-12))

    def test_parameter_partition_excludes_bn_and_bias_from_decay(self):
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 7)
        groups, names = split_parameter_groups(model, BACKBONE_LR, HEAD_LR, 5e-4)
        self.assertEqual(len(groups), 4)
        self.assertIn("fc.weight", names["head_decay"])
        self.assertIn("fc.bias", names["head_no_decay"])
        self.assertFalse(any("bn" in name for name in names["backbone_decay"]))
        self.assertFalse(
            any(name.endswith("bias") for name in names["backbone_decay"] + names["head_decay"])
        )
        all_names = [name for bucket in names.values() for name in bucket]
        self.assertCountEqual(all_names, [name for name, _ in model.named_parameters()])

    def test_transform_order_and_imagenet_normalization(self):
        train, validation = make_transforms()
        self.assertEqual(
            [type(item) for item in train.transforms],
            [
                transforms.Resize,
                transforms.RandomCrop,
                transforms.RandomHorizontalFlip,
                transforms.ToTensor,
                transforms.Normalize,
            ],
        )
        self.assertEqual(
            [type(item) for item in validation.transforms],
            [transforms.Resize, transforms.CenterCrop, transforms.ToTensor, transforms.Normalize],
        )
        self.assertEqual(tuple(train.transforms[-1].mean), IMAGENET_MEAN)
        self.assertEqual(tuple(train.transforms[-1].std), IMAGENET_STD)

    def test_machine_readable_spec_matches_constants(self):
        spec = json.loads((HERE / "hard_label_v1_protocol.json").read_text(encoding="utf-8"))
        self.assertEqual(spec["protocol"], "hard_label_v1")
        self.assertEqual(spec["optimization"]["total_optimizer_updates"], TOTAL_UPDATES)
        self.assertEqual(spec["optimization"]["backbone_initial_lr"], BACKBONE_LR)
        self.assertEqual(spec["optimization"]["head_initial_lr"], HEAD_LR)
        self.assertEqual(tuple(spec["input"]["normalization_mean"]), IMAGENET_MEAN)
        self.assertEqual(tuple(spec["input"]["normalization_std"]), IMAGENET_STD)
        self.assertEqual(spec["student"]["weights_file_sha256"], IMAGENET_V1_FILE_SHA256)


if __name__ == "__main__":
    unittest.main()
