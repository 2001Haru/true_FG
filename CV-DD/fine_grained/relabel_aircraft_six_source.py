"""Generate matched uncompressed/compressed FKD for two-source parent slots."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from paired_source_dataset import PairedEpochSampler, PairedSoftSaveDataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relabel.utils_fkd import mix_aug
from fine_grained.rescore_fkd_views import load_teacher, sha256


def atomic(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n"); os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--reference-fkd", required=True, type=Path)
    p.add_argument("--compressed-fkd", required=True, type=Path)
    p.add_argument("--teacher", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=400); p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--workers", type=int, default=8); p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dataset = PairedSoftSaveDataset(a.manifest, schedule_seed=a.seed); sampler = PairedEpochSampler(dataset, a.seed)
    loader = DataLoader(dataset, batch_size=a.batch_size, sampler=sampler, num_workers=a.workers,
                        persistent_workers=a.workers > 0, pin_memory=True)
    teacher = load_teacher(a.teacher, 100); mix_args = SimpleNamespace(mode="fkd_save", mix_type="cutmix", cutmix=1.0, mixup=.8)
    manifest_hash = sha256(a.manifest); teacher_hash = sha256(a.teacher)
    common = {"status": "running", "protocol": "aircraft_six_source_paired_soft_v1", "epochs": a.epochs,
              "batch_size": a.batch_size, "parents": 300, "sources": 600, "storage_ipc": 3,
              "paired_source_manifest": str(a.manifest.resolve()), "paired_source_manifest_sha256": manifest_hash,
              "source_schedule": "recorded per view in FKD payload; student never re-derives from epoch",
              "metadata_entries": ["coords", "flip", "mix_index", "mix_lambda", "mix_bbox", "logits", "subsource_index", "parent_index"],
              "view_order": "select subsource -> inverse-map/decode to 224 -> flip -> CutMix",
              "rrc": False, "teacher_path": str(a.teacher.resolve()), "teacher_sha256": teacher_hash,
              "teacher_mode": "train", "teacher_forward_split": [10, 10], "temperature": 20,
              "seed": a.seed, "sampler": "PairedEpochSampler stable per-epoch parent permutation"}
    for mode, root in (("reference", a.reference_fkd), ("compressed", a.compressed_fkd)):
        payload = dict(common, image_mode=mode, fkd_path=str(root.resolve())); atomic(payload, root / "relabel_manifest.json")
    for epoch in range(a.epochs):
        dataset.set_epoch(epoch)
        dirs = []
        for root in (a.reference_fkd, a.compressed_fkd):
            directory = root / f"epoch_{epoch}"; directory.mkdir(parents=True, exist_ok=True); dirs.append(directory)
        batches = 0
        for batch, data in enumerate(loader):
            ref, comp, target, flip, coords, sources, parents = data
            ref = ref.cuda(non_blocking=True); comp = comp.cuda(non_blocking=True)
            mixed_ref, mix_index, mix_lam, mix_bbox = mix_aug(ref.clone(), mix_args)
            mixed_comp, _, _, _ = mix_aug(comp.clone(), SimpleNamespace(mode="fkd_load", mix_type="cutmix", cutmix=1.0, mixup=.8),
                                          mix_index, mix_lam, mix_bbox)
            outputs = []
            for images in (mixed_ref, mixed_comp):
                logits = torch.cat((teacher(images[:10]), teacher(images[10:])), 0).half().cpu(); outputs.append(logits)
            metadata = [coords, flip, mix_index, mix_lam, mix_bbox]
            torch.save(metadata + [outputs[0], sources, parents], dirs[0] / f"batch_{batch}.tar")
            torch.save(metadata + [outputs[1], sources, parents], dirs[1] / f"batch_{batch}.tar")
            batches += 1
        if batches != 15: raise RuntimeError(f"epoch {epoch}: {batches}")
        print(f"epoch={epoch} batches={batches}", flush=True)
    for mode, root in (("reference", a.reference_fkd), ("compressed", a.compressed_fkd)):
        files = list(root.glob("epoch_*/batch_*.tar"))
        if len(files) != 6000: raise RuntimeError(f"{mode} files={len(files)}")
        payload = json.loads((root / "relabel_manifest.json").read_text()); payload["status"] = "complete"; payload["batch_files"] = 6000
        atomic(payload, root / "relabel_manifest.json")
    print(json.dumps({"status": "complete", "reference": str(a.reference_fkd), "compressed": str(a.compressed_fkd)}))


if __name__ == "__main__": main()
