"""Cross-family BN-tax gate using layer input |log variance ratios|."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models
from torchvision.transforms import functional as TF

from paired_source_dataset import AIRCRAFT_MEAN, AIRCRAFT_STD, PairedSourceIndex


STAGES = ("stem", "layer1", "layer2", "layer3", "layer4")


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i]); result = [0.] * len(values); start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]: end += 1
        value = (start + end - 1) / 2
        for position in range(start, end): result[order[position]] = value
        start = end
    return result


def spearman(left, right):
    left, right = ranks(left), ranks(right); ml, mr = np.mean(left), np.mean(right)
    numerator = sum((x - ml) * (y - mr) for x, y in zip(left, right))
    denominator = math.sqrt(sum((x - ml) ** 2 for x in left) * sum((y - mr) ** 2 for y in right))
    return numerator / denominator if denominator else 0.


def stage(name):
    return "stem" if name == "bn1" else name.split(".", 1)[0]


def tensor(image):
    return TF.normalize(TF.pil_to_tensor(image).float().div_(255), AIRCRAFT_MEAN, AIRCRAFT_STD)


def paired_entries(condition):
    if condition["kind"] == "paired":
        variant = PairedSourceIndex(condition["path"], condition["mode"])
        clear = PairedSourceIndex(condition["path"], "reference")
        return [(clear.load(parent, source), variant.load(parent, source))
                for parent in range(len(variant.rows)) for source in range(variant.sources_per_parent)]
    variant_root, reference_root = Path(condition["path"]), Path(condition["reference_root"])
    entries = []
    for variant_path in sorted(variant_root.glob("*/*")):
        if not variant_path.is_file(): continue
        original = reference_root / variant_path.parent.name / variant_path.name
        if not original.is_file():
            matches = sorted((reference_root / variant_path.parent.name).glob(variant_path.stem + ".*"))
            if len(matches) != 1: raise FileNotFoundError((original, matches))
            original = matches[0]
        with Image.open(original) as handle: clear = handle.convert("RGB")
        with Image.open(variant_path) as handle: variant = handle.convert("RGB")
        entries.append((clear, variant))
    return entries


def load_model(path, device):
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model = models.resnet18(weights=None); model.fc = nn.Linear(model.fc.in_features, 100)
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def distances(model, entries, batch_size, device):
    modules = {name: module for name, module in model.named_modules() if isinstance(module, nn.BatchNorm2d)}
    accumulators = {name: {side: {"sum": torch.zeros(module.num_features, dtype=torch.float64, device=device),
                                        "sumsq": torch.zeros(module.num_features, dtype=torch.float64, device=device),
                                        "count": 0}
                           for side in ("clear", "variant")} for name, module in modules.items()}
    captured = {}; handles = []
    for name, module in modules.items():
        handles.append(module.register_forward_pre_hook(
            lambda _, inputs, key=name: captured.__setitem__(key, inputs[0].detach())))
    for offset in range(0, len(entries), batch_size):
        chunk = entries[offset:offset + batch_size]
        clear = torch.stack([tensor(pair[0]) for pair in chunk]).to(device)
        variant = torch.stack([tensor(pair[1]) for pair in chunk]).to(device)
        for flip in (False, True):
            if flip: clear_view, variant_view = clear.flip(-1), variant.flip(-1)
            else: clear_view, variant_view = clear, variant
            joined = torch.cat((clear_view, variant_view)); model(joined); size = len(chunk)
            for name, activation in captured.items():
                for side, value in (("clear", activation[:size]), ("variant", activation[size:])):
                    value = value.double(); target = accumulators[name][side]
                    target["sum"] += value.sum((0, 2, 3)); target["sumsq"] += value.square().sum((0, 2, 3))
                    target["count"] += value.shape[0] * value.shape[2] * value.shape[3]
    for handle in handles: handle.remove()
    by_stage = {name: [] for name in STAGES}; modules_out = {}
    for name, sides in accumulators.items():
        variance = {}
        for side, acc in sides.items():
            mean = acc["sum"] / acc["count"]
            variance[side] = (acc["sumsq"] / acc["count"] - mean.square()).clamp_min(0)
        channel = ((variance["variant"] + 1e-6).log() - (variance["clear"] + 1e-6).log()).abs()
        value = float(channel.mean()); by_stage[stage(name)].append(value)
        modules_out[name] = {"abs_logvar_ratio_mean": value, "channels": len(channel)}
    stages = {name: float(np.mean(values)) for name, values in by_stage.items()}
    stages["five_stage_equal_mean"] = float(np.mean([stages[name] for name in STAGES]))
    stages["layer1_4_equal_mean"] = float(np.mean([stages[name] for name in STAGES[1:]]))
    return {"stages": stages, "modules": modules_out}


def evaluate(rows, names):
    taxes = [rows[name]["tax_top1"] for name in names]; output = {}
    metrics = list(STAGES) + ["five_stage_equal_mean", "layer1_4_equal_mean"]
    for metric in metrics:
        values = [rows[name]["mean_across_students"][metric] for name in names]
        mapping = dict(zip(names, values)); rho = spearman(values, taxes)
        bbox_pass = mapping["bbox"] > mapping["k4_f1_v1"]
        output[metric] = {"spearman": rho, "bbox_above_k4_f1_v1": bbox_pass,
                          "passes_same_rule": bool(rho >= .7 and bbox_pass), "values": mapping}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", required=True, type=Path)
    parser.add_argument("--student", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(); device = torch.device("cuda")
    conditions = json.loads(args.conditions.read_text())
    expected = ["k2_f1", "k3_f1", "k4_f1_v1", "k4_f1_v2", "bbox",
                "object_decoded", "plain", "cam", "random_protected"]
    if [row["name"] for row in conditions] != expected or len(args.student) != 3:
        raise RuntimeError("preregistered conditions/students changed")
    models_list = [load_model(path, device) for path in args.student]; rows = {}
    for condition in conditions:
        entries = paired_entries(condition)
        per_student = [distances(model, entries, args.batch_size, device) for model in models_list]
        means = {metric: float(np.mean([item["stages"][metric] for item in per_student]))
                 for metric in per_student[0]["stages"]}
        rows[condition["name"]] = {"tax_top1": float(condition["tax_top1"]), "images": len(entries),
                                    "optimization_contaminated": condition["name"] == "k4_f1_v2",
                                    "mean_across_students": means,
                                    "per_student_stages": [item["stages"] for item in per_student]}
        print(condition["name"], json.dumps(rows[condition["name"]]), flush=True)
    primary_names = expected
    clean_names = [name for name in expected if name != "k4_f1_v2"]
    evaluation9 = evaluate(rows, primary_names); evaluation8 = evaluate(rows, clean_names)
    output = {
        "status": "complete", "protocol": "aircraft_abs_logvar_tax_gate_v1",
        "definition": "for every BN input module, dataset-level per-channel variance over paired clear/variant identity+flip views; mean channel |log((var_variant+1e-6)/(var_clear+1e-6))|; stage is equal-module mean",
        "preregistration": {
            "primary_metric": "stem", "primary_conditions": primary_names,
            "pass_rule": "nine-condition Spearman>=0.7 and BBox>k4-F1-v1",
            "k4_f1_v2": "included in primary table and marked optimization-contaminated; eight-condition exclusion is sensitivity only",
            "other_stages": "descriptive localization only; no post-hoc metric selection",
        },
        "rows": rows, "evaluation_9": evaluation9, "evaluation_8_excluding_k4_f1_v2": evaluation8,
        "primary_pass": evaluation9["stem"]["passes_same_rule"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"primary": evaluation9["stem"], "all_stage_rhos": {key: value["spearman"] for key, value in evaluation9.items()},
                      "excluding_v2_stem": evaluation8["stem"]}, indent=2))


if __name__ == "__main__":
    main()
