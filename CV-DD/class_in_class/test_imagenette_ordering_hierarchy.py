import unittest

import numpy as np
import torch

from prepare_imagenette_ordering_hierarchy import apply_mapping, mapping_for_order


class OrderingTests(unittest.TestCase):
    def test_template_order_receives_descending_non_target_values(self):
        probability = torch.tensor([.35, .03, .01, .20, .08, .06, .04, .02, .11, .10])
        target = 0
        destination = [7, 2, 5, 4, 1, 9, 6, 8, 3]
        mapping = mapping_for_order(probability, target, destination)
        changed = apply_mapping(probability, mapping)
        assigned = changed[torch.tensor(destination)]
        self.assertTrue(torch.all(assigned[:-1] >= assigned[1:]))
        self.assertEqual(float(changed[target]), float(probability[target]))

    def test_inverse_control_is_exactly_equidistant(self):
        probability = torch.tensor([.31, .19, .14, .10, .08, .06, .05, .03, .025, .015])
        target = 3
        destination = [9, 1, 7, 0, 8, 5, 2, 6, 4]
        mapping = mapping_for_order(probability, target, destination)
        inverse = np.argsort(mapping).tolist()
        arm_a = apply_mapping(probability, mapping)
        arm_aprime = apply_mapping(probability, inverse)
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(arm_a - probability),
                torch.linalg.vector_norm(arm_aprime - probability),
            )
        )
        self.assertTrue(torch.equal(torch.sort(arm_a).values, torch.sort(probability).values))
        self.assertTrue(torch.equal(torch.sort(arm_aprime).values, torch.sort(probability).values))


if __name__ == "__main__":
    unittest.main()
