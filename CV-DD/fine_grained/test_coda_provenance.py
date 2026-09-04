"""Contract tests for CoDA per-source and per-generated-image provenance."""

import hashlib
import importlib.util
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CoDAProvenanceTest(unittest.TestCase):
    def test_random_real_selection_is_deterministic_and_nested(self):
        preparer = load_module("random_real_preparer", "prepare_random_real_fg.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            for class_id in range(2):
                class_dir = data / "train" / f"{class_id:03d}"
                class_dir.mkdir(parents=True)
                for image_id in range(6):
                    Image.new("RGB", (8, 8), color=(class_id, image_id, 0)).save(
                        class_dir / f"{image_id}.png"
                    )
            chosen = {}
            for ipc in (1, 3, 5):
                output = root / f"ipc{ipc}"
                manifest = root / f"ipc{ipc}.json"
                argv = [
                    "prepare",
                    "--data-dir", str(data),
                    "--output-dir", str(output),
                    "--manifest", str(manifest),
                    "--dataset-name", "fixture",
                    "--classes", "2",
                    "--ipc", str(ipc),
                    "--selection-seed", "0",
                    "--link-mode", "copy",
                ]
                with mock.patch.object(sys, "argv", argv):
                    preparer.main()
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                chosen[ipc] = {row["source_path"] for row in payload["images"]}
                self.assertEqual(payload["selected_images"], 2 * ipc)
                with mock.patch.object(sys, "argv", argv):
                    preparer.main()
            self.assertLessEqual(chosen[1], chosen[3])
            self.assertLessEqual(chosen[3], chosen[5])

    def test_hard_label_result_audit_is_explicit_and_backward_compatible(self):
        auditor = load_module("result_auditor", "audit_result.py")
        payload = {
            "best_top1": 50.0,
            "training_target": "hard_coarse_label",
            "num_classes": 2,
            "validation_images": 2,
            "primary_metric": "native_top1",
            "native_top1_at_best_checkpoint": 50.0,
            "per_class": [
                {"class_id": 0, "correct": 1, "total": 1, "accuracy": 100.0},
                {"class_id": 1, "correct": 0, "total": 1, "accuracy": 0.0},
            ],
        }
        with self.assertRaisesRegex(RuntimeError, "expected fkd_soft_label"):
            auditor.audit_payload(payload, classes=2, validation_images=2)
        self.assertEqual(
            auditor.audit_payload(
                payload,
                classes=2,
                validation_images=2,
                expected_training_target="hard_coarse_label",
            ),
            50.0,
        )
        soft_payload = dict(payload, training_target="fkd_soft_label")
        self.assertEqual(
            auditor.audit_payload(soft_payload, classes=2, validation_images=2),
            50.0,
        )

    def test_cluster_and_generation_audits_preserve_image_lineage(self):
        cluster_auditor = load_module("cluster_auditor", "audit_coda_clusters.py")
        generation_auditor = load_module("generation_auditor", "audit_coda_fg.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            cluster = root / "cluster"
            synthetic = root / "generated"
            for class_id in range(2):
                folder = f"{class_id:03d}"
                (data / "train" / folder).mkdir(parents=True)
                (synthetic / folder).mkdir(parents=True)
                (data / "train" / folder / "source.jpg").write_bytes(b"source")
                Image.new("RGB", (224, 224), color=(class_id, 0, 0)).save(
                    synthetic / folder / "0.png"
                )
            cluster.mkdir()
            with (cluster / "1_n_5_s_2_saved_clusters_0.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        0: np.zeros((1, 65536), dtype=np.float16),
                        1: np.zeros((1, 65536), dtype=np.float16),
                    },
                    handle,
                )
            provenance_path = cluster / "1_n_5_s_2_image_provenance_0.jsonl"
            provenance_rows = []
            for class_id in range(2):
                folder = f"{class_id:03d}"
                provenance_rows.append(
                    {
                        "schema_version": 1,
                        "feature_space": "vae",
                        "class_id": class_id,
                        "class_folder": folder,
                        "class_prompt": folder,
                        "source_index_within_class": 0,
                        "source_path": str((data / "train" / folder / "source.jpg").resolve()),
                        "initial_hdbscan_cluster_count": 0 if class_id == 0 else 1,
                        "initial_hdbscan_label": -1 if class_id == 0 else 0,
                        "initial_hdbscan_is_noise": class_id == 0,
                        "initial_hdbscan_membership_probability": 0.0 if class_id == 0 else 1.0,
                        "initial_hdbscan_outlier_score": 1.0 if class_id == 0 else 0.0,
                        "final_disposition": "retained_final_cluster_member",
                        "final_cluster_id": 0,
                        "final_cluster_origin": "kmeans_outliers" if class_id == 0 else "hdbscan_initial",
                        "final_cluster_size": 1,
                        "selected_as_representative": True,
                        "representative_slot": 0,
                        "representative_origin": "kmeans_outliers" if class_id == 0 else "hdbscan_initial",
                        "representative_selection": "nearest_real_image_to_postprocessed_center",
                        "generated_image_relative_path": f"{folder}/0.png",
                    }
                )
            provenance_path.write_text(
                "".join(json.dumps(row) + "\n" for row in provenance_rows), encoding="utf-8"
            )
            cluster_audit = root / "cluster_audit.json"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "audit",
                    "--cluster-dir", str(cluster),
                    "--data-dir", str(data),
                    "--classes", "2",
                    "--ipc", "1",
                    "--feature-space", "vae",
                    "--n-neighbors", "5",
                    "--min-cluster-size", "2",
                    "--output", str(cluster_audit),
                ],
            ):
                cluster_auditor.main()
            cluster_payload = json.loads(cluster_audit.read_text(encoding="utf-8"))
            self.assertEqual(cluster_payload["overall"]["initial_noise_images"], 1)
            self.assertEqual(cluster_payload["overall"]["selected_initial_noise"], 1)

            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {"feature_space": "vae", "generation_seed": 0, "generation_gpu_count": 1}
                ),
                encoding="utf-8",
            )
            trace = root / "trace.json"
            trace.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "feature_space": "vae",
                        "generation_seed": 0,
                        "generation_gpu_count": 1,
                        "images": [
                            {
                                "class_id": class_id,
                                "class_folder": f"{class_id:03d}",
                                "representative_slot": 0,
                                "prompt": f"{class_id:03d}",
                                "gpu_id": 0,
                                "image_seed": 1000 + class_id,
                                "output_path": str(
                                    (synthetic / f"{class_id:03d}" / "0.png").resolve()
                                ),
                            }
                            for class_id in range(2)
                        ],
                    }
                ),
                encoding="utf-8",
            )
            generation_auditor.get_dataset = lambda _: SimpleNamespace(
                name="fixture", classes=2
            )
            generated_provenance = root / "generated_provenance.jsonl"
            generation_audit = root / "generation_audit.json"
            with mock.patch.object(
                sys,
                "argv",
                [
                    "audit",
                    "--dataset-name", "fixture",
                    "--data-dir", str(data),
                    "--synthetic-dir", str(synthetic),
                    "--ipc", "1",
                    "--generation-config", str(config),
                    "--cluster-audit", str(cluster_audit),
                    "--generation-trace", str(trace),
                    "--generated-provenance-output", str(generated_provenance),
                    "--output", str(generation_audit),
                ],
            ):
                generation_auditor.main()
            generation_payload = json.loads(generation_audit.read_text(encoding="utf-8"))
            self.assertEqual(generation_payload["generated_images_from_initial_noise"], 1)
            self.assertEqual(len(generated_provenance.read_text(encoding="utf-8").splitlines()), 2)
            self.assertEqual(
                generation_payload["generated_image_provenance_sha256"],
                digest(generated_provenance),
            )


if __name__ == "__main__":
    unittest.main()
