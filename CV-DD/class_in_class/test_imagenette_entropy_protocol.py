import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from imagenette_entropy_protocol import (
    DeterministicReleasedView,
    average_tie_percentiles,
    build_student,
    gumbel_noise,
    identity_seed,
    lr_for_epoch,
    state_dict_sha256,
)


class EntropyProtocolTest(unittest.TestCase):
    def test_tie_percentiles_span_zero_to_one(self):
        self.assertEqual(average_tie_percentiles([1.0, 2.0, 3.0]), [0.0, 0.5, 1.0])
        self.assertEqual(average_tie_percentiles([1.0, 1.0, 2.0]), [0.25, 0.25, 1.0])
        self.assertEqual(average_tie_percentiles([1.0, 2.0, 2.0]), [0.0, 0.75, 0.75])

    def test_gumbel_is_identity_keyed(self):
        a = gumbel_noise(0, "train/a/x.jpg")
        self.assertEqual(a, gumbel_noise(0, "train/a/x.jpg"))
        self.assertNotEqual(a, gumbel_noise(1, "train/a/x.jpg"))

    def test_released_view_is_deterministic_and_seed_sensitive(self):
        array = np.arange(300 * 400 * 3, dtype=np.int64).reshape(300, 400, 3) % 256
        image = Image.fromarray(array.astype(np.uint8))
        transform = DeterministicReleasedView()
        seed = identity_seed("test", 42, 1, "x")
        first = transform(image, seed)
        second = transform(image, seed)
        other = transform(image, seed + 1)
        self.assertEqual(tuple(first.shape), (3, 256, 256))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))

    def test_lr_changes_after_declared_epoch(self):
        self.assertEqual(lr_for_epoch(1), 1e-2)
        self.assertEqual(lr_for_epoch(1333), 1e-2)
        self.assertEqual(lr_for_epoch(1334), 2e-3)
        self.assertEqual(lr_for_epoch(1666), 2e-3)
        self.assertEqual(lr_for_epoch(1667), 4e-4)
        self.assertEqual(lr_for_epoch(2000), 4e-4)

    def test_soft1_equals_ce_when_teacher_is_one_hot(self):
        logits = torch.tensor([[0.2, -0.1, 0.7], [1.0, 0.5, -0.2]])
        targets = torch.tensor([2, 0])
        one_hot = torch.nn.functional.one_hot(targets, 3).float()
        soft = -(one_hot * torch.log_softmax(logits, dim=1)).sum(1).mean()
        hard = torch.nn.functional.cross_entropy(logits, targets)
        self.assertTrue(torch.equal(soft, hard))

    def test_released_student_has_groupnorm_and_no_batchnorm(self):
        model = build_student()
        group_norms = [module for module in model.modules() if isinstance(module, torch.nn.GroupNorm)]
        batch_norms = [module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)]
        self.assertTrue(group_norms)
        self.assertFalse(batch_norms)
        self.assertTrue(all(module.num_groups == module.num_channels for module in group_norms))

    def test_student_initialization_is_seed_pairable(self):
        torch.manual_seed(42)
        hard = build_student()
        torch.manual_seed(42)
        soft = build_student()
        self.assertEqual(
            state_dict_sha256(hard.state_dict()),
            state_dict_sha256(soft.state_dict()),
        )


if __name__ == "__main__":
    unittest.main()
