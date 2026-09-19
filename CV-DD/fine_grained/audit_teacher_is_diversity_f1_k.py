"""Teacher-IS and zero-cost terminal-loss Diversity for Aircraft F1 k arms."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from audit_cmmd_vendi_f1_k import AIRCRAFT_MEAN, AIRCRAFT_STD, PairedDecodedDataset, normalize


LINE = re.compile(r"TRAIN Iter (\d+):.*?loss = ([0-9.eE+-]+)")


def load_teacher(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "model" in state: state = state["model"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model = models.resnet18(weights=None); model.fc = nn.Linear(model.fc.in_features, 100)
    model.load_state_dict(state)
    return model.cuda().eval()


@torch.no_grad()
def teacher_is(dataset, teacher, batch_size, workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        persistent_workers=workers > 0, pin_memory=True)
    probabilities, labels = [], []
    for images, target in loader:
        images = images.cuda(non_blocking=True).float().div_(255)
        logits = teacher(normalize(images, AIRCRAFT_MEAN, AIRCRAFT_STD))
        probabilities.append(F.softmax(logits.float(), dim=1).cpu()); labels.append(target)
    probabilities = torch.cat(probabilities).double(); labels = torch.cat(labels)
    marginal = probabilities.mean(0)
    kl = (probabilities * (probabilities.clamp_min(1e-300).log() -
                           marginal.clamp_min(1e-300).log()[None])).sum(1)
    conditional_entropy = -(probabilities * probabilities.clamp_min(1e-300).log()).sum(1)
    marginal_entropy = -(marginal * marginal.clamp_min(1e-300).log()).sum()
    return {"teacher_is": float(kl.mean().exp()), "mean_kl": float(kl.mean()),
            "marginal_entropy": float(marginal_entropy),
            "mean_conditional_entropy": float(conditional_entropy.mean()),
            "teacher_top1": float((probabilities.argmax(1) == labels).double().mean() * 100),
            "mean_true_class_probability": float(probabilities[torch.arange(len(labels)), labels].mean()),
            "samples": len(labels)}


def parse_log(path):
    values = {}
    for match in LINE.finditer(path.read_text(errors="replace")):
        values[int(match.group(1))] = float(match.group(2))
    if set(values) != set(range(400)): raise RuntimeError((path, len(values), min(values), max(values)))
    return {"logged_final": values[399], "logged_tail10_mean": float(np.mean([values[index] for index in range(390, 400)])),
            "objective_final": 2 * values[399], "objective_tail10_mean": 2 * float(np.mean([values[index] for index in range(390, 400)])),
            "epochs": len(values), "scale_note": "logged loss is divided by gradient_accumulation_steps=2 before AverageMeter; objective values multiply by 2"}


def cached_view_teacher_is(root):
    probability_sum = torch.zeros(100, dtype=torch.float64); entropy_sum = 0.; count = 0; files = 0
    for epoch in range(400):
        directory = root / f"epoch_{epoch}"
        paths = sorted(directory.glob("batch_*.tar"), key=lambda path: int(path.stem.split("_")[1]))
        if len(paths) != 15: raise RuntimeError((root, epoch, len(paths)))
        for path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            logits = payload[5].float(); probability = F.softmax(logits, dim=1).double()
            probability_sum += probability.sum(0)
            entropy_sum += float(-(probability * probability.clamp_min(1e-300).log()).sum())
            count += len(probability); files += 1
    marginal = probability_sum / count
    marginal_entropy = float(-(marginal * marginal.clamp_min(1e-300).log()).sum())
    conditional_entropy = entropy_sum / count
    return {"teacher_is": float(np.exp(marginal_entropy - conditional_entropy)),
            "mean_kl": marginal_entropy - conditional_entropy,
            "marginal_entropy": marginal_entropy, "mean_conditional_entropy": conditional_entropy,
            "views": count, "batch_files": files, "teacher_mode": "cached train-mode BSSL",
            "view": "formal flip+CutMix training views"}


def summarize_logs(paths):
    rows = [parse_log(path) for path in paths]
    output = {"per_student": rows}
    for key in ("logged_final", "logged_tail10_mean", "objective_final", "objective_tail10_mean"):
        values = [row[key] for row in rows]
        output[key] = {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1)), "values": values}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", action="append", required=True, help="name=manifest")
    parser.add_argument("--logs", action="append", required=True, help="name=log42,log43,log44")
    parser.add_argument("--fkd", action="append", required=True, help="name=FKD root")
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    specs = dict(value.split("=", 1) for value in args.spec)
    logs = {}
    for value in args.logs:
        name, paths = value.split("=", 1); logs[name] = [Path(path) for path in paths.split(",")]
    fkds = {name: Path(path) for name, path in (value.split("=", 1) for value in args.fkd)}
    if set(specs) != set(logs) or set(specs) != set(fkds): raise RuntimeError((set(specs), set(logs), set(fkds)))
    teacher = load_teacher(args.teacher); groups = {}
    for name, manifest in specs.items():
        groups[name] = {"teacher_is_t1_actual_views": cached_view_teacher_is(fkds[name]),
                        "teacher_is_t1_source_eval": teacher_is(PairedDecodedDataset(manifest), teacher,
                                                      args.batch_size, args.workers),
                        "diversity_terminal_train_loss": summarize_logs(logs[name])}
        print(name, json.dumps(groups[name]), flush=True)
    output = {"status": "complete", "protocol": "aircraft_f1_k_teacher_is_diversity_v1",
              "teacher_is": {"definition": "exp(E KL(p_T1(y|x)||mean_x p_T1(y|x)))",
                             "primary": "cached actual flip+CutMix views from train-mode BSSL Teacher",
                             "sensitivity": "each decoded source once with eval-mode Teacher"},
              "diversity": {"definition": "terminal Student training KL on the actual augmented epoch-399 trajectory",
                            "primary": "objective_final", "sensitivity": "objective_tail10_mean",
                            "source": "existing Soft-v2 logs; zero new training"},
              "groups": groups}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
