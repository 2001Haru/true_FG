"""Re-evaluate the ten-arm proxy gate on actual flip+CutMix training views."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from audit_six_source_feasibility_attribution import spearman
from paired_source_dataset import PairedSourceIndex


MEAN = (.4865, .5177, .5425)
STD = (.2124, .2051, .2375)
SINGLE_ARMS = ("isotropic158", "vertical112", "horizontal112",
               "bbox_retarget112", "attention_retarget112")
SIX_ARMS = ("uniform", "bbox", "background_hsmooth", "object_decoded", "background_decoded")
SIX_MODES = {"uniform": "uniform", "bbox": "compressed",
             "background_hsmooth": "background_hsmooth", "object_decoded": "object_decoded",
             "background_decoded": "background_decoded"}
LOSSES = {
    "single": {"isotropic158": 6.0706070607, "vertical112": 4.2504250425,
               "horizontal112": 12.2012201220, "bbox_retarget112": 3.2903290329,
               "attention_retarget112": 4.4604460446},
    "six": {"uniform": 3.3903390339, "bbox": 2.2902290229,
            "background_hsmooth": 2.3202320232, "object_decoded": 1.8801880188,
            "background_decoded": .4600460046},
}


def load_model(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model = models.resnet18(weights=None); model.fc = nn.Linear(model.fc.in_features, 100)
    model.load_state_dict(state)
    return model.cuda().eval()


def image_tensor(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), np.uint8).copy()).permute(2, 0, 1)


def transform_batch(images, coords, flips):
    rows = []
    for image, coord, flip in zip(images, coords, flips):
        coord = torch.as_tensor(coord).float()
        if torch.allclose(coord, torch.tensor([0., 0., 1., 1.]), rtol=0, atol=1e-7):
            view = image
        else:
            top, left = float(coord[0]) * 224, float(coord[1]) * 224
            height, width = float(coord[2]) * 224, float(coord[3]) * 224
            view = TF.resized_crop(image, top, left, height, width, (224, 224),
                                   interpolation=InterpolationMode.BILINEAR,
                                   antialias=False)
        if bool(flip): view = TF.hflip(view)
        rows.append(view.float().div(255))
    batch = torch.stack(rows).cuda(non_blocking=True)
    mean = torch.tensor(MEAN, device="cuda")[None, :, None, None]
    std = torch.tensor(STD, device="cuda")[None, :, None, None]
    return (batch - mean) / std


def cutmix(images, mix_index, bbox):
    output = images.clone(); x1, y1, x2, y2 = map(int, bbox)
    output[:, :, y1:y2, x1:x2] = images[mix_index.long().cuda(), :, y1:y2, x1:x2]
    keep = 1. - ((x2 - x1) * (y2 - y1)) / float(224 * 224)
    return output, keep


@torch.no_grad()
def update(models_, original, variants, labels, mix_index, bbox, accumulators):
    original, keep = cutmix(original, mix_index, bbox)
    variants = {name: cutmix(value, mix_index, bbox)[0] for name, value in variants.items()}
    labels = labels.cuda(); donor = labels[mix_index.long().cuda()]
    for model_index, model in enumerate(models_):
        original_logits = model(original)
        original_pred = original_logits.argmax(1)
        original_accuracy = keep * (original_pred == labels).float() + (1 - keep) * (original_pred == donor).float()
        original_log = F.log_softmax(original_logits / 20, dim=1); original_prob = original_log.exp()
        for name, value in variants.items():
            logits = model(value); logp = F.log_softmax(logits / 20, dim=1)
            kl = (original_prob * (original_log - logp)).sum(1)
            pred = logits.argmax(1)
            accuracy = keep * (pred == labels).float() + (1 - keep) * (pred == donor).float()
            row = accumulators[name][model_index]
            row["kl_sum"] += float(kl.sum())
            row["original_accuracy_sum"] += float(original_accuracy.sum())
            row["variant_accuracy_sum"] += float(accuracy.sum())
            row["count"] += len(labels)


def evaluate(accumulators):
    output = {}
    for arm, rows in accumulators.items():
        individual = []
        for row in rows:
            count = row["count"]
            individual.append({
                "view_kl20": row["kl_sum"] / count,
                "original_weighted_top1": 100 * row["original_accuracy_sum"] / count,
                "variant_weighted_top1": 100 * row["variant_accuracy_sum"] / count,
                "affinity_drop": 100 * (row["original_accuracy_sum"] - row["variant_accuracy_sum"]) / count,
                "views": count,
            })
        output[arm] = {
            "view_kl20": float(np.mean([row["view_kl20"] for row in individual])),
            "affinity_drop": float(np.mean([row["affinity_drop"] for row in individual])),
            "individual_students": individual,
        }
    return output


def blank(arms, models_):
    return {arm: [{"kl_sum": 0., "original_accuracy_sum": 0., "variant_accuracy_sum": 0., "count": 0}
                  for _ in models_] for arm in arms}


def single_views(args, models_, epochs):
    manifest = json.loads(args.single_manifest.read_text())
    records = {str(Path(row["source_path"]).resolve()): row for row in manifest["records"]}
    samples = []
    for class_id, directory in enumerate(sorted(path for path in args.single_root.iterdir() if path.is_dir())):
        for path in sorted(directory.iterdir()): samples.append((path.resolve(), class_id))
    if len(samples) != 300 or any(str(path) not in records for path, _ in samples):
        raise RuntimeError("single-arm source mapping failed")
    originals = torch.stack([image_tensor(path) for path, _ in samples])
    variants = {arm: torch.stack([image_tensor(records[str(path)]["outputs"][arm]["decoded"])
                                  for path, _ in samples]) for arm in SINGLE_ARMS}
    labels_all = torch.tensor([label for _, label in samples])
    accumulators = blank(SINGLE_ARMS, models_); generator = torch.Generator().manual_seed(42)
    for epoch in range(400):
        order = torch.randperm(300, generator=generator)
        if epoch not in epochs: continue
        for batch in range(15):
            ids = order[batch * 20:(batch + 1) * 20]
            config = torch.load(args.single_fkd / f"epoch_{epoch}/batch_{batch}.tar",
                                map_location="cpu", weights_only=False)
            coords, flips, mix_index, _, bbox = config[:5]
            original = transform_batch(originals[ids], coords, flips)
            arm_views = {arm: transform_batch(value[ids], coords, flips) for arm, value in variants.items()}
            update(models_, original, arm_views, labels_all[ids], mix_index, bbox, accumulators)
        print(f"single epoch={epoch}", flush=True)
    return evaluate(accumulators)


def six_views(args, models_, epochs):
    reference = PairedSourceIndex(args.six_manifest, "reference")
    indices = {arm: PairedSourceIndex(args.six_manifest, SIX_MODES[arm]) for arm in SIX_ARMS}
    reference_cache = [[image_tensor(row["sources"][source]["reference_path"]) for source in range(2)]
                       for row in reference.rows]
    variant_cache = {arm: [[TF.pil_to_tensor(index.load(parent, source)) for source in range(2)]
                           for parent in range(len(index.rows))] for arm, index in indices.items()}
    accumulators = blank(SIX_ARMS, models_)
    for epoch in sorted(epochs):
        for batch in range(15):
            config = torch.load(args.six_fkd / f"epoch_{epoch}/batch_{batch}.tar",
                                map_location="cpu", weights_only=False)
            coords, flips, mix_index, _, bbox, _, sources, parents = config
            original_images = torch.stack([reference_cache[int(parent)][int(source)]
                                           for parent, source in zip(parents, sources)])
            original = transform_batch(original_images, coords, flips)
            arm_views = {}
            for arm, cache in variant_cache.items():
                images = torch.stack([cache[int(parent)][int(source)] for parent, source in zip(parents, sources)])
                arm_views[arm] = transform_batch(images, coords, flips)
            labels = torch.tensor([reference.targets[int(parent)] for parent in parents])
            update(models_, original, arm_views, labels, mix_index, bbox, accumulators)
        print(f"six epoch={epoch}", flush=True)
    return evaluate(accumulators)


def gate(scores, metric):
    evaluation = {}; passed = True
    for setting, arms in (("single", SINGLE_ARMS), ("six", SIX_ARMS)):
        values = [scores[setting][arm][metric] for arm in arms]
        losses = [LOSSES[setting][arm] for arm in arms]
        rho = spearman(values, losses); evaluation[setting] = {"spearman": rho, "scores": values,
                                                               "student_losses": losses}
        passed &= rho >= .9 - 1e-12
    single, six = scores["single"], scores["six"]
    get = lambda rows, arm: rows[arm][metric]
    single_order = (get(single, "bbox_retarget112") < get(single, "attention_retarget112") and
                    get(single, "horizontal112") == max(get(single, arm) for arm in SINGLE_ARMS))
    six_order = get(six, "background_decoded") < get(six, "object_decoded")
    six_values = [get(six, arm) for arm in SIX_ARMS]
    adjacent = abs(sorted(six_values).index(get(six, "background_hsmooth")) -
                   sorted(six_values).index(get(six, "bbox"))) <= 1
    close = abs(get(six, "background_hsmooth") - get(six, "bbox")) <= .25 * (max(six_values) - min(six_values))
    critical = {"single_bbox_lt_attention_and_horizontal_worst": single_order,
                "six_background_lt_object": six_order,
                "six_smooth_approximately_bbox": adjacent and close}
    return {"metric": metric, "settings": evaluation, "critical_pairs": critical,
            "passes": bool(passed and all(critical.values()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-manifest", required=True, type=Path)
    parser.add_argument("--single-root", required=True, type=Path)
    parser.add_argument("--single-fkd", required=True, type=Path)
    parser.add_argument("--six-manifest", required=True, type=Path)
    parser.add_argument("--six-fkd", required=True, type=Path)
    parser.add_argument("--student", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sampled-epochs", type=int, default=40)
    args = parser.parse_args()
    models_ = [load_model(path) for path in args.student]
    epochs = set(np.linspace(0, 399, args.sampled_epochs).round().astype(int).tolist())
    scores = {"single": single_views(args, models_, epochs), "six": six_views(args, models_, epochs)}
    gates = {metric: gate(scores, metric) for metric in ("view_kl20", "affinity_drop")}
    output = {"status": "complete", "protocol": "ten_arm_actual_view_proxy_v1",
              "view": "recorded fullframe/flip/CutMix; identical metadata for original and decoded image",
              "affinity_definition": "original-view weighted CutMix top1 minus decoded-view weighted CutMix top1; larger is worse",
              "p5_definition": "KL_T20(R0 Student(original actual view) || R0 Student(decoded actual view))",
              "epochs": sorted(epochs), "views_per_arm_per_student": len(epochs) * 300,
              "students": [str(path) for path in args.student], "scores": scores,
              "gates": gates, "passing_metrics": [name for name, value in gates.items() if value["passes"]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"passing_metrics": output["passing_metrics"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
