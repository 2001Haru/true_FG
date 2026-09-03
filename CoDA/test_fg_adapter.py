"""CPU contract tests for the fine-grained CoDA adapter."""

import tempfile
import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fg_prompts import (
    AIRCRAFT_VARIANTS,
    STANFORD_CARS_CLASSES,
    effective_umap_dimensions,
    prompts_for_classes,
)
class FineGrainedCoDATest(unittest.TestCase):
    def test_canonical_prompt_counts(self):
        self.assertEqual(len(AIRCRAFT_VARIANTS), 100)
        self.assertEqual(len(set(AIRCRAFT_VARIANTS)), 100)
        self.assertEqual(len(STANFORD_CARS_CLASSES), 196)
        self.assertEqual(len(set(STANFORD_CARS_CLASSES)), 196)

    def test_prompt_alignment(self):
        cub = prompts_for_classes("cub", ["001.Black_footed_Albatross"])
        self.assertEqual(cub["001.Black_footed_Albatross"], "Black footed Albatross")
        aircraft = prompts_for_classes("aircraft", [f"{i:03d}" for i in range(100)])
        self.assertEqual(aircraft["000"], "707-320")
        self.assertEqual(aircraft["099"], "Yak-42")
        cars = prompts_for_classes("cars", [f"{i:03d}" for i in range(196)])
        self.assertEqual(cars["000"], "AM General Hummer SUV 2000")
        self.assertEqual(cars["195"], "smart fortwo Convertible 2012")

    def test_umap_dimensions_are_capped_for_small_classes(self):
        self.assertEqual(effective_umap_dimensions(29), 27)
        self.assertEqual(effective_umap_dimensions(30), 28)
        self.assertEqual(effective_umap_dimensions(66), 50)

    @unittest.skipUnless(
        importlib.util.find_spec("sklearn") and importlib.util.find_spec("hdbscan"),
        "CoDA clustering dependencies are not installed",
    )
    def test_small_class_all_noise_uses_algorithm2_outlier_fallback(self):
        from postprocess import hdbscan_post

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                spec="cars",
                IPC=5,
                min_cluster_size=5,
                min_samples=3,
                num_seed_candidates=3,
                cluster_detial=False,
                cluster_logger=False,
                next_new_id=0,
            )
            features = np.random.RandomState(0).normal(size=(24, 4))
            result = hdbscan_post(
                args,
                M=0,
                initial_centers=[],
                cluster_labels=np.full(24, -1),
                X=features,
                log_file_path=str(Path(directory) / "unused.log"),
                clusterer=None,
            )
            self.assertEqual(len(result), 5)
            self.assertEqual(sum(item["size"] for item in result.values()), 24)


if __name__ == "__main__":
    unittest.main()
