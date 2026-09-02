"""Fast CPU contract tests for the two FD2 interpretations."""

import unittest

import torch

from fd2_standard_components import (
    CAL,
    compose_recovery_loss,
    fine_grained_characteristic_loss,
    normalized_l2,
    previous_ids_in_group,
    update_feature_centers,
)


class FD2SemanticsTest(unittest.TestCase):
    def test_cal_tensor_contract_and_reload(self):
        cal = CAL(num_classes=3, attention_maps=2).eval()
        features = torch.randn(2, 512, 7, 7)
        outputs = cal(features)
        self.assertEqual(tuple(outputs[0].shape), (2, 3))
        self.assertEqual(tuple(outputs[1].shape), (2, 3))
        self.assertEqual(tuple(outputs[2].shape), (2, 1024))
        self.assertEqual(tuple(outputs[3].shape), (2, 1, 7, 7))
        self.assertEqual(tuple(outputs[4].shape), (2, 2, 7, 7))
        reloaded = CAL(num_classes=3, attention_maps=2).eval()
        reloaded.load_state_dict(cal.state_dict())
        with torch.no_grad():
            self.assertTrue(torch.equal(cal(features)[0], reloaded(features)[0]))

    def test_ns4_ipc5_is_four_plus_one(self):
        self.assertEqual(previous_ids_in_group(0), ())
        self.assertEqual(previous_ids_in_group(1), (0,))
        self.assertEqual(previous_ids_in_group(2), (0, 1))
        self.assertEqual(previous_ids_in_group(3), (0, 1, 2))
        self.assertEqual(previous_ids_in_group(4), ())

    def test_duplicate_label_prototype_updates_are_distinct(self):
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0, 0])
        released = torch.zeros(2, 2)
        literal = torch.zeros(2, 2)
        update_feature_centers(released, features, labels, "released_semantics")
        update_feature_centers(literal, features, labels, "paper_literal")
        self.assertFalse(torch.equal(released, literal))
        self.assertTrue(torch.allclose(released[0], torch.tensor([0.0, 0.025])))
        expected = torch.tensor([2.0**-0.5, 2.0**-0.5]) * 0.05
        self.assertTrue(torch.allclose(literal[0], expected))

    def test_recovery_objectives_are_audibly_different(self):
        values = {
            "ce_backbone": torch.tensor(2.0),
            "ce_cal": torch.tensor(4.0),
            "bn_backbone": torch.tensor(3.0),
            "bn_cal": torch.tensor(5.0),
            "feature_loss": torch.tensor(0.5),
            "similarity": torch.tensor(0.25),
            "cal_ratio": 0.3,
            "r_bn": 0.1,
        }
        released, _ = compose_recovery_loss("released_semantics", **values)
        literal, _ = compose_recovery_loss("paper_literal", **values)
        # released = Lcls + BN(backbone+CAL) + .9 LF + .1 LS
        self.assertAlmostEqual(float(released), 3.875, places=6)
        # literal = (CEbb+BNbb) + Lcls + .8 LF + .2 LS
        self.assertAlmostEqual(float(literal), 5.35, places=6)

    def test_vectorized_feature_loss_matches_direct_definition(self):
        torch.manual_seed(7)
        features = torch.randn(4, 6)
        centers = torch.randn(5, 6)
        labels = torch.tensor([0, 1, 2, 3])
        intra = normalized_l2(features, centers[labels]).mean()
        inter = torch.stack(
            [
                normalized_l2(feature.unsqueeze(0), centers[torch.arange(5) != label]).mean()
                for feature, label in zip(features, labels)
            ]
        ).mean()
        direct = 0.5 * intra + 0.5 * (1.0 - inter)
        vectorized = fine_grained_characteristic_loss(features, labels, centers, beta=0.5)
        self.assertTrue(torch.allclose(vectorized, direct, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
