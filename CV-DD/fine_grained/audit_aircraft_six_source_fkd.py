"""Audit all parent/subsource and augmentation metadata in paired six-source FKD."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from paired_source_dataset import PairedSourceIndex


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path); p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--compressed", required=True, type=Path); p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(); index = PairedSourceIndex(a.manifest, "reference")
    counts = np.zeros((300, 2), dtype=np.int64); mismatch = {k: 0 for k in ("coords", "flip", "mix_index", "mix_lambda", "mix_bbox", "source", "parent")}
    digest = hashlib.sha256(); batches = examples = 0
    for epoch in range(400):
        seen = []
        for batch in range(15):
            left = torch.load(a.reference / f"epoch_{epoch}/batch_{batch}.tar", map_location="cpu", weights_only=False)
            right = torch.load(a.compressed / f"epoch_{epoch}/batch_{batch}.tar", map_location="cpu", weights_only=False)
            if len(left) != 8 or len(right) != 8: raise RuntimeError("payload length")
            for pos, key in zip((0, 1, 2, 3, 4, 6, 7), mismatch):
                equal = torch.equal(left[pos], right[pos]) if torch.is_tensor(left[pos]) else left[pos] == right[pos]
                mismatch[key] += int(not equal)
            for parent, source in zip(left[7].tolist(), left[6].tolist()):
                if source != index.source_index(parent, epoch): raise RuntimeError(f"schedule mismatch e{epoch} p{parent}")
                counts[parent, source] += 1; seen.append(parent); digest.update(f"{epoch}:{parent}:{source}\n".encode())
            batches += 1; examples += len(left[7])
        if sorted(seen) != list(range(300)): raise RuntimeError(f"epoch {epoch} parent coverage")
    result = {"status": "complete" if not sum(mismatch.values()) else "failed", "batches": batches, "examples": examples,
              "mismatch_batches": mismatch, "parent_epoch_coverage": "exactly once per epoch",
              "source_exposure_min": int(counts.min()), "source_exposure_max": int(counts.max()),
              "expected_source_exposure": 200, "schedule_sha256": digest.hexdigest(),
              "student_replay": "payload parent/subsource is authoritative; runtime asserts sampler parent matches cached parent"}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "complete" or counts.min() != 200 or counts.max() != 200: raise SystemExit(1)


if __name__ == "__main__": main()
