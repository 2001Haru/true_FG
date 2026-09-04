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
