import json
import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18


VALIDATE = Path(__file__).resolve().parents[1] / "validate"
sys.path.insert(0, str(VALIDATE))
from utils_validate import get_finetune_parameter_groups  # noqa: E402


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, bias=True)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 10)


class StandardV2ProtocolTest(unittest.TestCase):
    def test_protocol_definition(self):
        path = Path(__file__).with_name("standard_v2_protocol.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["version"], "v2")
        self.assertEqual(payload["recovery_iterations"], {
            "CUB_imsize224": 4000, "A_imsize224": 4000, "SC_imsize224": 4000,
        })
        self.assertEqual(payload["optimizer"]["backbone_initial_lr"], 1e-4)
        self.assertEqual(payload["optimizer"]["classifier_initial_lr"], 1e-3)
        self.assertEqual(payload["scheduler"]["t_max"], 400)
        self.assertEqual(payload["scheduler"]["eta_min"], 0.0)

    def test_parameter_groups(self):
        model = TinyStudent()
        groups = get_finetune_parameter_groups(model, 1e-4, 1e-3, 1e-5)
        rows = {group["group_name"]: group for group in groups}
        self.assertEqual(set(rows), {
            "backbone_decay", "backbone_no_decay", "head_decay", "head_no_decay",
        })
        self.assertEqual(rows["backbone_decay"]["parameter_names"], ["conv.weight"])
        self.assertEqual(rows["backbone_no_decay"]["parameter_names"], [
            "conv.bias", "bn.weight", "bn.bias",
        ])
        self.assertEqual(rows["head_decay"]["parameter_names"], ["fc.weight"])
        self.assertEqual(rows["head_no_decay"]["parameter_names"], ["fc.bias"])
        self.assertEqual(rows["backbone_decay"]["weight_decay"], 1e-5)
        self.assertEqual(rows["backbone_no_decay"]["weight_decay"], 0.0)
        self.assertEqual(rows["head_decay"]["lr"], 1e-3)

    def test_torchvision_resnet18_head_and_bn_grouping(self):
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 200)
        rows = {
            group["group_name"]: group
            for group in get_finetune_parameter_groups(model, 1e-4, 1e-3, 1e-5)
        }
        self.assertEqual(rows["head_decay"]["parameter_names"], ["fc.weight"])
        self.assertEqual(rows["head_no_decay"]["parameter_names"], ["fc.bias"])
        self.assertIn("bn1.weight", rows["backbone_no_decay"]["parameter_names"])
        self.assertIn("layer4.1.bn2.bias", rows["backbone_no_decay"]["parameter_names"])
        self.assertNotIn("bn1.weight", rows["backbone_decay"]["parameter_names"])

    def test_epoch_cosine_semantics(self):
        model = TinyStudent()
        groups = get_finetune_parameter_groups(model, 1e-4, 1e-3, 1e-5)
        optimizer = torch.optim.AdamW(
            groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=400, eta_min=0.0,
        )
        used = []
        for epoch in range(400):
            used.append([group["lr"] for group in optimizer.param_groups])
            optimizer.step()
            scheduler.step()
        self.assertEqual(used[0][0], 1e-4)
        expected_last = 1e-4 * (1 + math.cos(math.pi * 399 / 400)) / 2
        self.assertGreater(used[-1][0], 0.0)
        self.assertTrue(math.isclose(used[-1][0], expected_last, rel_tol=1e-10))
        self.assertTrue(all(abs(value) < 1e-15 for value in scheduler.get_last_lr()))


if __name__ == "__main__":
    unittest.main()
