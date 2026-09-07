import unittest

import torch
import torch.nn.functional as F

from train_imagenette_entropy_selection import mixed_hard_soft1_loss
from prepare_imagenette_nontarget_permutations import fixed_derangement


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

    def test_fixed_derangement_preserves_only_target_identity(self):
        mapping, _ = fixed_derangement("train/00003/example.jpg", 3)
        self.assertEqual(sorted(mapping), list(range(10)))
        self.assertEqual(mapping[3], 3)
        self.assertTrue(all(mapping[index] != index for index in range(10) if index != 3))
        self.assertEqual(mapping, fixed_derangement("train/00003/example.jpg", 3)[0])

    def test_derangement_preserves_requested_probability_invariants(self):
        target = 3
        mapping, _ = fixed_derangement("train/00003/example.jpg", target)
        probability = torch.softmax(torch.arange(10, dtype=torch.float32), dim=0)
        permuted = probability[torch.tensor(mapping)]
        onehot = F.one_hot(torch.tensor(target), num_classes=10).float()
        self.assertEqual(float(probability[target]), float(permuted[target]))
        self.assertEqual(float(probability.max()), float(permuted.max()))
        self.assertTrue(torch.allclose(torch.sort(probability).values, torch.sort(permuted).values))
        self.assertTrue(torch.allclose(-(probability * probability.log()).sum(), -(permuted * permuted.log()).sum()))
        self.assertTrue(torch.allclose((probability - onehot).abs().sum(), (permuted - onehot).abs().sum()))
        self.assertTrue(torch.allclose((probability - onehot).square().sum(), (permuted - onehot).square().sum()))


if __name__ == "__main__":
    unittest.main()
