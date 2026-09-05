"""End-to-end fixture test for controlled five-arm DINO IPC1 selection."""

import hashlib
import json
import math
import pickle
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SELECTOR = HERE / "prepare_dino_fourway_ipc1.py"
SHELL_RANDOM = HERE / "prepare_shell_random_extension.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DinoFourwaySelectionTest(unittest.TestCase):
    def test_geometry_selection_provenance_and_idempotency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            cache_dir = root / "cache"
            model_dir = root / "dino"
            output_root = root / "selection"
            cache_dir.mkdir()
            model_dir.mkdir()
            paths = defaultdict(list)
            features = defaultdict(list)
            for class_id in range(3):
                class_dir = data_dir / "train" / f"class_{class_id:02d}"
                class_dir.mkdir(parents=True)
                for image_id in range(20):
                    image = class_dir / f"image_{image_id:02d}.jpg"
                    image.write_bytes(f"{class_id}-{image_id}".encode())
                    theta = (image_id - 9.5) * 0.02
                    feature = np.zeros(768, dtype=np.float32)
                    feature[class_id] = math.cos(theta)
                    feature[(class_id + 1) % 3] = math.sin(theta)
                    paths[class_id].append(str(image.resolve()))
                    features[class_id].append(feature)
            with (cache_dir / "original_features_cache.pkl_0").open("wb") as handle:
                pickle.dump({"paths": paths, "features": features}, handle)
            feature_audit = root / "feature_audit.json"
            feature_audit.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "feature_space": "dinov2",
                        "classes": 3,
                        "images": 60,
                        "feature_dimension": 768,
                        "cache_dir": str(cache_dir.resolve()),
                    }
                ),
                encoding="utf-8",
            )
            for name, content in (
                ("model.safetensors", b"model"),
                ("config.json", b"{}"),
                ("preprocessor_config.json", b"{}"),
            ):
                (model_dir / name).write_bytes(content)
            generation_config = root / "generation_config.json"
            generation_config.write_text(
                json.dumps(
                    {
                        "status": "frozen",
                        "dataset": "fixture",
                        "classes": 3,
                        "feature_space": "dinov2",
                        "data_dir": str(data_dir.resolve()),
                        "dino_model_root": str(model_dir.resolve()),
                        "dino_model_sha256": sha256(model_dir / "model.safetensors"),
                        "dino_config_sha256": sha256(model_dir / "config.json"),
                        "dino_preprocessor_sha256": sha256(
                            model_dir / "preprocessor_config.json"
                        ),
                        "clustering_feature_encoder": (
                            "DINOv2-base final normalized CLS token (768D), "
                            "Resize256+CenterCrop224"
                        ),
                        "source_sha256": {
                            "CoDA/get_features.py": sha256(REPO_ROOT / "CoDA" / "get_features.py")
                        },
                    }
                ),
                encoding="utf-8",
            )
            audit = root / "selection_audit.json"
            command = [
                sys.executable,
                str(SELECTOR),
                "--data-dir",
                str(data_dir),
                "--cache-dir",
                str(cache_dir),
                "--feature-audit",
                str(feature_audit),
                "--generation-config",
                str(generation_config),
                "--dino-model-root",
                str(model_dir),
                "--repo-root",
                str(REPO_ROOT),
                "--output-root",
                str(output_root),
                "--audit-output",
                str(audit),
                "--dataset-name",
                "fixture",
                "--classes",
                "3",
                "--link-mode",
                "copy",
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            first = audit.read_bytes()
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(first, audit.read_bytes())
            payload = json.loads(first)
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(len(payload["images"]), 60)
            self.assertTrue(all(row["shell_images"] > 0 for row in payload["class_summaries"]))
            for arm in (
                "centroid",
                "rival_facing_edge",
                "outward_edge",
                "edge_high_margin",
                "random_rseed0",
                "random_rseed1",
                "random_rseed2",
            ):
                manifest = json.loads(
                    (output_root / "manifests" / f"{arm}.json").read_text(encoding="utf-8")
                )
                self.assertEqual(manifest["selection_images"], 3)
                self.assertEqual(len(manifest["images"]), 3)
            for class_id in range(3):
                centroid = payload["class_summaries"][class_id]["selected_indices"]["centroid"]
                rival_facing = payload["class_summaries"][class_id]["selected_indices"][
                    "rival_facing_edge"
                ]
                outward = payload["class_summaries"][class_id]["selected_indices"][
                    "outward_edge"
                ]
                self.assertNotEqual(centroid, rival_facing)
                self.assertNotEqual(rival_facing, outward)
            self.assertFalse(payload["shell"]["prototype_correctness_filter"])
            frozen_geometry = audit.read_bytes()
            extension_audit = root / "shell_random_audit.json"
            extension_command = [
                sys.executable,
                str(SHELL_RANDOM),
                "--geometry-audit",
                str(audit),
                "--selection-base",
                str(output_root),
                "--extension-audit",
                str(extension_audit),
                "--dataset-name",
                "fixture",
                "--classes",
                "3",
                "--link-mode",
                "copy",
            ]
            completed = subprocess.run(extension_command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(audit.read_bytes(), frozen_geometry)
            extension = json.loads(extension_audit.read_text(encoding="utf-8"))
            self.assertEqual(extension["status"], "complete")
            self.assertFalse(extension["uses_rival_similarity"])
            self.assertEqual(len(extension["images"]), 9)
            self.assertTrue(all(row["in_edge_shell"] for row in extension["images"]))
            first_extension = extension_audit.read_bytes()
            completed = subprocess.run(extension_command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(extension_audit.read_bytes(), first_extension)
            self.assertEqual(audit.read_bytes(), frozen_geometry)
            for seed in range(3):
                manifest = json.loads(
                    (
                        output_root / "manifests" / f"shell_random_rseed{seed}.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(manifest["experiment"], "dino_sixarm_ipc1")
                self.assertEqual(manifest["selection_method"], "shell_random")
                self.assertEqual(manifest["selection_images"], 3)


if __name__ == "__main__":
    unittest.main()
