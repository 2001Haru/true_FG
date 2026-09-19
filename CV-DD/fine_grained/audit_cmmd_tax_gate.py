"""Preregistered CMMD gate for the nine non-zero Aircraft BN-tax arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from audit_cmmd_common_views import CommonSchedule, CommonViewDataset
from audit_cmmd_vendi_f1_k import extract, unbiased_cmmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", required=True, type=Path)
    parser.add_argument("--geometry-fkd", required=True, type=Path)
    parser.add_argument("--reference-features", required=True, type=Path)
    parser.add_argument("--clip-source-root", required=True, type=Path)
    parser.add_argument("--clip-download-root", type=Path, default=Path("/root/.cache/clip"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    conditions = json.loads(args.conditions.read_text())
    expected = ["k2_f1", "k3_f1", "k4_f1_v1", "k4_f1_v2", "bbox",
                "object_decoded", "plain", "cam", "random_protected"]
    if [row["name"] for row in conditions] != expected:
        raise RuntimeError("condition order/content differs from preregistration")
    epochs = sorted(set(np.linspace(0, 399, 32).round().astype(int).tolist()))
    schedule = CommonSchedule(args.geometry_fkd, epochs)
    sensitivity_indices = []
    for parent in range(300):
        chosen = np.random.default_rng(20260919 + parent).choice(len(epochs), size=2, replace=False)
        sensitivity_indices.extend(int(position) * 300 + parent for position in sorted(chosen))
    sensitivity_indices = torch.tensor(sorted(sensitivity_indices), dtype=torch.long)
    sys.path.insert(0, str(args.clip_source_root.resolve()))
    import clip
    clip_model, _ = clip.load("ViT-L/14@336px", device="cuda", jit=False,
                              download_root=str(args.clip_download_root))
    clip_model = clip_model.float().eval()
    references = {name: torch.from_numpy(np.load(args.reference_features / f"{name}.npz")["clip"])
                  for name in ("test", "train", "r0")}
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = {}
    for condition in conditions:
        if condition["kind"] == "paired":
            dataset = CommonViewDataset(schedule, epochs, "off", paired_manifest=Path(condition["path"]),
                                        paired_mode=condition["mode"])
        elif condition["kind"] == "imagefolder":
            dataset = CommonViewDataset(schedule, epochs, "off", image_root=Path(condition["path"]))
        else:
            raise RuntimeError(condition["kind"])
        features, _, labels = extract(dataset, clip_model, None, args.batch_size, args.workers, False,
                                      args.output_root / "features" / f"{condition['name']}.npz")
        if len(features) != 9600 or sorted(torch.bincount(labels, minlength=100).tolist()) != [96] * 100:
            raise RuntimeError((condition["name"], len(features)))
        scores = {}
        for count, subset in ((600, features[sensitivity_indices]), (9600, features)):
            scores[f"n{count}"] = {ref: unbiased_cmmd(subset, value) for ref, value in references.items()}
        rows[condition["name"]] = {"tax_top1": float(condition["tax_top1"]), "cmmd": scores}
        print(condition["name"], json.dumps(rows[condition["name"]]), flush=True)
    evaluations = {}
    for count in (600, 9600):
        for reference in references:
            tax = [rows[name]["tax_top1"] for name in expected]
            score = [rows[name]["cmmd"][f"n{count}"][reference] for name in expected]
            rho, pvalue = spearmanr(score, tax)
            bbox_pass = rows["bbox"]["cmmd"][f"n{count}"][reference] > rows["k4_f1_v1"]["cmmd"][f"n{count}"][reference]
            evaluations[f"n{count}_{reference}"] = {
                "spearman": float(rho), "two_sided_pvalue": float(pvalue),
                "bbox_above_k4_f1_v1": bool(bbox_pass),
                "passes_same_rule": bool(rho >= .7 and bbox_pass),
            }
    primary = evaluations["n9600_test"]
    output = {
        "status": "complete", "protocol": "aircraft_cmmd_nonzero_tax_gate_v1",
        "preregistration": {
            "conditions": expected, "primary": "N=9600 CMMD-test, RRC-off/full-frame+flip, CutMix-off",
            "pass_rule": "Spearman>=0.7 across nine non-zero-tax conditions AND BBox CMMD > k4-F1-v1 CMMD",
            "sensitivity": "N=600 and train/R0 references do not affect the primary verdict",
        },
        "geometry_schedule_sha256": schedule.sha256,
        "n600_indices_sha256": hashlib.sha256(sensitivity_indices.numpy().tobytes()).hexdigest(),
        "rows": rows, "evaluations": evaluations,
        "primary_pass": bool(primary["passes_same_rule"]),
    }
    (args.output_root / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
