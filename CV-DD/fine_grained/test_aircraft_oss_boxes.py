import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_aircraft_oss_boxes import box_metrics, mask_box


class TestAircraftOssBoxes(unittest.TestCase):
    def test_identical_box(self):
        result = box_metrics([1, 2, 11, 12], [1, 2, 11, 12], 20, 20)
        self.assertAlmostEqual(result["iou"], 1.0)
        self.assertAlmostEqual(result["area_ratio"], 1.0)
        self.assertAlmostEqual(result["center_error_diagonal"], 0.0)

    def test_partial_box(self):
        result = box_metrics([0, 0, 10, 10], [5, 0, 15, 10], 20, 20)
        self.assertAlmostEqual(result["iou"], 1 / 3)
        self.assertAlmostEqual(result["official_coverage"], .5)

    def test_mask_box_continuous_xyxy(self):
        mask = np.zeros((8, 9), dtype=bool); mask[2:6, 3:8] = True
        self.assertEqual(mask_box(mask), [3.0, 2.0, 8.0, 6.0])
        self.assertIsNone(mask_box(np.zeros((2, 2), dtype=bool)))


if __name__ == "__main__": unittest.main()
