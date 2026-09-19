"""Validate actual-view P5 against the preregistered 12-condition BN-tax gate."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from audit_view_level_affinity import cutmix, image_tensor, load_model, transform_batch
from paired_source_dataset import PairedSourceIndex


def ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index]); output = [0.] * len(values); start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]: end += 1
        value = (start + end - 1) / 2
        for position in range(start, end): output[order[position]] = value
        start = end
    return output


def spearman(left, right):
    left, right = ranks(left), ranks(right); ml, mr = np.mean(left), np.mean(right)
    numerator = sum((x - ml) * (y - mr) for x, y in zip(left, right))
    denominator = math.sqrt(sum((x - ml) ** 2 for x in left) * sum((y - mr) ** 2 for y in right))
    return numerator / denominator if denominator else 0.


@torch.no_grad()
def batch_kl(models_, original, variant, mix_index, bbox):
    original = cutmix(original, mix_index, bbox)[0]
    variant = cutmix(variant, mix_index, bbox)[0]
    output = []
    for model in models_:
        joined = model(torch.cat((original, variant))); size = len(original)
        log_original = F.log_softmax(joined[:size] / 20, 1)
        log_variant = F.log_softmax(joined[size:] / 20, 1)
        output.append(float((log_original.exp() * (log_original - log_variant)).sum()))
    return output


def paired_condition(condition, models_, epochs):
    reference = PairedSourceIndex(condition["manifest"], "reference")
    variant = PairedSourceIndex(condition["manifest"], condition["mode"])
    original_cache = [[image_tensor(row["sources"][source]["reference_path"])
                       for source in range(reference.sources_per_parent)] for row in reference.rows]
    variant_cache = [[torch.from_numpy(np.asarray(variant.load(parent, source), np.uint8).copy()).permute(2, 0, 1)
                      for source in range(variant.sources_per_parent)] for parent in range(len(variant.rows))]
    sums = [0.] * len(models_); count = 0
    for epoch in sorted(epochs):
        for batch in range(15):
            config = torch.load(Path(condition["fkd"]) / f"epoch_{epoch}/batch_{batch}.tar",
                                map_location="cpu", weights_only=False)
            if len(config) != 8: raise RuntimeError((condition["name"], len(config)))
            coords, flips, mix_index, _, bbox, _, sources, parents = config
            original_images = torch.stack([original_cache[int(parent)][int(source)]
                                           for parent, source in zip(parents, sources)])
            variant_images = torch.stack([variant_cache[int(parent)][int(source)]
                                          for parent, source in zip(parents, sources)])
            original = transform_batch(original_images, coords, flips)
            changed = transform_batch(variant_images, coords, flips)
            values = batch_kl(models_, original, changed, mix_index, bbox)
            sums = [left + right for left, right in zip(sums, values)]; count += len(original)
        print(f"{condition['name']} epoch={epoch}", flush=True)
    return [value / count for value in sums], count


def imagefolder_condition(condition, models_, epochs):
    variant_root, reference_root = Path(condition["path"]), Path(condition["reference_root"])
    samples = []
    for directory in sorted(path for path in variant_root.iterdir() if path.is_dir()):
        for changed in sorted(path for path in directory.iterdir() if path.is_file()):
            original = reference_root / directory.name / changed.name
            if not original.is_file():
                matches = sorted((reference_root / directory.name).glob(changed.stem + ".*"))
                if len(matches) != 1: raise FileNotFoundError((changed, original, matches))
                original = matches[0]
            samples.append((original, changed))
    if len(samples) != 300: raise RuntimeError((condition["name"], len(samples)))
    originals = torch.stack([image_tensor(original) for original, _ in samples])
    variants = torch.stack([image_tensor(changed) for _, changed in samples])
    generator = torch.Generator().manual_seed(42); sums = [0.] * len(models_); count = 0
    for epoch in range(400):
        order = torch.randperm(300, generator=generator)
        if epoch not in epochs: continue
        for batch in range(15):
            ids = order[batch * 20:(batch + 1) * 20]
            config = torch.load(Path(condition["fkd"]) / f"epoch_{epoch}/batch_{batch}.tar",
                                map_location="cpu", weights_only=False)
            if len(config) not in (6, 7): raise RuntimeError((condition["name"], len(config)))
            coords, flips, mix_index, _, bbox = config[:5]
            original = transform_batch(originals[ids], coords, flips)
            changed = transform_batch(variants[ids], coords, flips)
            values = batch_kl(models_, original, changed, mix_index, bbox)
            sums = [left + right for left, right in zip(sums, values)]; count += len(original)
        print(f"{condition['name']} epoch={epoch}", flush=True)
    return [value / count for value in sums], count


def evaluate(rows, conditions, key):
    names = [condition["name"] for condition in conditions if condition.get("validation", True)]
    taxes = [rows[name]["tax_top1"] for name in names]
    values = [rows[name][key] for name in names]
    mapping = dict(zip(names, values))
    order_pass = mapping["k2_f1"] < mapping["k3_f1"] < mapping["k4_f1_v1"] < mapping["k4_f1_v2"]
    rho = spearman(values, taxes)
    return {"spearman": rho, "order_pass": order_pass, "passes": bool(rho >= .7 and order_pass),
            "values": mapping}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", required=True, type=Path)
    parser.add_argument("--student", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sampled-epochs", type=int, default=40)
    args = parser.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    conditions = json.loads(args.conditions.read_text())["conditions"]
    models_ = [load_model(path) for path in args.student]
    epochs = set(np.linspace(0, 399, args.sampled_epochs).round().astype(int).tolist())
    rows = {}
    for condition in conditions:
        if condition["kind"] == "zero":
            rows[condition["name"]] = {"tax_top1": condition["tax_top1"], "view_p5": 0.,
                                        "per_student": [0.] * len(models_), "views": 0}
            continue
        if condition["kind"] == "paired": values, count = paired_condition(condition, models_, epochs)
        elif condition["kind"] == "imagefolder": values, count = imagefolder_condition(condition, models_, epochs)
        else: raise ValueError(condition["kind"])
        rows[condition["name"]] = {"tax_top1": condition["tax_top1"],
                                    "view_p5": float(np.mean(values)), "per_student": values, "views": count}
        gc.collect(); torch.cuda.empty_cache()
    mean_evaluation = evaluate(rows, conditions, "view_p5")
    individual = []
    for index in range(len(models_)):
        projected = {name: {**row, "student_p5": row["per_student"][index]} for name, row in rows.items()}
        individual.append(evaluate(projected, conditions, "student_p5"))
    output = {"status": "complete", "protocol": "actual_view_p5_tax_12_conditions_v1",
              "definition": "KL_T20(clear actual flip+CutMix view || candidate actual flip+CutMix view); exact per-condition FKD metadata",
              "epochs": sorted(epochs), "views_per_nonzero_condition_per_student": len(epochs) * 300,
              "students": [str(path) for path in args.student], "rows": rows,
              "evaluation": mean_evaluation, "per_student_evaluation": individual,
              "passing_predictors": ["view_p5"] if mean_evaluation["passes"] else []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"evaluation": mean_evaluation, "per_student": individual}, indent=2))


if __name__ == "__main__":
    main()
