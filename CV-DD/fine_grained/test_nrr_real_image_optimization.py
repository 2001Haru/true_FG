import importlib.util
import math
import unittest
from pathlib import Path

import torch
from torch import nn


PATH = Path(__file__).with_name("nrr_real_image_optimization.py")
SPEC = importlib.util.spec_from_file_location("nrr_real_image_optimization", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NRRPixelOptimizationTests(unittest.TestCase):
    def test_cosine_lr_actual_update_endpoints(self):
        self.assertEqual(MODULE.cosine_lr(1, 2000, 0.01, 1e-4), 0.01)
        self.assertEqual(MODULE.cosine_lr(2000, 2000, 0.01, 1e-4), 1e-4)
        self.assertGreater(MODULE.cosine_lr(1000, 2000, 0.01, 1e-4), 1e-4)

    def test_projection_and_hard_protection(self):
        initial = torch.tensor([[[[0.02, 0.5], [0.98, 0.4]]] * 3])
        candidate = torch.tensor([[[[-1.0, 0.9], [2.0, 0.1]]] * 3])
        protected = torch.tensor([[[False, True], [False, False]]])
        result = MODULE.project_pixels(candidate, initial, protected, 32 / 255)
        self.assertTrue(torch.equal(result[:, :, 0, 1], initial[:, :, 0, 1]))
        self.assertLessEqual(float((result - initial).abs().max()), 32 / 255 + 1e-7)
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_topk_mask_has_exact_area_and_stable_ties(self):
        values = torch.zeros(224, 224)
        mask = MODULE.topk_mask(values, MODULE.PROTECTED_PIXELS)
        self.assertEqual(int(mask.sum()), MODULE.PROTECTED_PIXELS)
        self.assertTrue(bool(mask.flatten()[0]))
        self.assertFalse(bool(mask.flatten()[-1]))

    def test_flip_stream_is_reproducible_and_arm_independent(self):
        first = MODULE.flip_flags(100, 42, 2, 901)
        second = MODULE.flip_flags(100, 42, 2, 901)
        other_update = MODULE.flip_flags(100, 42, 2, 902)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other_update))

    def test_bn_hook_uses_input_population_variance(self):
        bn = nn.BatchNorm2d(2)
        bn.running_mean.copy_(torch.tensor([0.5, -0.5]))
        bn.running_var.copy_(torch.tensor([2.0, 3.0]))
        bn.eval()
        hook = MODULE.BNInputHook(bn)
        values = torch.tensor([[[[1.0, 2.0]], [[3.0, 7.0]]], [[[5.0, 8.0]], [[11.0, 13.0]]]])
        bn(values)
        mean = values.mean((0, 2, 3))
        variance = values.var((0, 2, 3), correction=0)
        expected = torch.linalg.vector_norm(mean - bn.running_mean) + torch.linalg.vector_norm(variance - bn.running_var)
        self.assertTrue(torch.allclose(hook.value, expected))
        hook.close()


if __name__ == "__main__":
    unittest.main()
