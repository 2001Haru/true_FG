import unittest

import torch
import torch.nn.functional as F

from train_imagenette_entropy_selection import mixed_hard_soft1_loss


class MixedLossTests(unittest.TestCase):
    def test_actual_batch_mean_matches_per_sample_definition(self):
        logits = torch.tensor(
            [[2.0, -1.0], [0.5, 0.1], [-0.2, 1.3], [1.0, 0.8]],
            requires_grad=True,
        )
        targets = torch.tensor([0, 0, 1, 1])
        soft_mask = torch.tensor([True, False, True, False])
        teacher = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        actual = mixed_hard_soft1_loss(logits, targets, soft_mask, teacher)
        log_prob = F.log_softmax(logits, dim=1)
        expected_rows = torch.stack(
            [
                -(teacher[0] * log_prob[0]).sum(),
                -log_prob[1, targets[1]],
                -(teacher[1] * log_prob[2]).sum(),
                -log_prob[3, targets[3]],
            ]
        )
        self.assertTrue(torch.allclose(actual, expected_rows.mean()))

    def test_rejects_mask_probability_mismatch(self):
        with self.assertRaises(RuntimeError):
            mixed_hard_soft1_loss(
                torch.zeros(3, 2),
                torch.zeros(3, dtype=torch.long),
                torch.tensor([True, False, True]),
                torch.ones(1, 2),
            )


if __name__ == "__main__":
    unittest.main()
