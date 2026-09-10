"""Score the full Aircraft train pool and materialize nested Teacher-score selections."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms


MEAN = (0.4865, 0.5177, 0.5425)
STD = (0.2124, 0.2051, 0.2375)
IPCS = (1, 3, 5)
METHODS = ("entropy_t20", "entropy_t1", "true_ce_t1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def entropy(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    log_probabilities = torch.log_softmax(logits.float() / temperature, dim=1)
    probabilities = log_probabilities.exp()
    return -(probabilities * log_probabilities).sum(dim=1)


def load_teacher(path: Path) -> nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


def link_selection(records: list[dict], output: Path) -> None:
    expected = set()
    for record in records:
        source = Path(record["source_path"])
        destination = output / record["class_folder"] / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected.add(destination.relative_to(output).as_posix())
        if destination.exists() or destination.is_symlink():
            if not destination.is_symlink() or destination.resolve() != source.resolve():
                raise RuntimeError(f"selection collision: {destination}")
        else:
            os.symlink(source.resolve(), destination)
    actual = {
        path.relative_to(output).as_posix()
        for path in output.glob("*/*") if path.is_file() or path.is_symlink()
    }
    if actual != expected:
        raise RuntimeError(f"selection tree mismatch at {output}: {len(actual)} != {len(expected)}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--skip-completed", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "selection_manifest.json"
    if args.skip_completed and manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("status") == "complete":
            print(f"Teacher-score selection already complete: {manifest_path}")
            return
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    source_root = (args.data_dir / "train").resolve()
    dataset = datasets.ImageFolder(
        source_root,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        ),
    )
    if len(dataset.classes) != 100 or len(dataset) != 6667:
        raise RuntimeError(f"unexpected Aircraft train set: classes={len(dataset.classes)} images={len(dataset)}")
    generator = torch.Generator().manual_seed(42)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0, generator=generator,
    )
    teacher = load_teacher(args.teacher)
    logits_parts = []
    labels_parts = []
    for images, labels in loader:
        logits_parts.append(teacher(images.cuda(non_blocking=True)).cpu().float())
        labels_parts.append(labels.clone())
    logits = torch.cat(logits_parts)
    labels = torch.cat(labels_parts)
    if logits.shape != (6667, 100):
        raise RuntimeError(f"unexpected logits shape: {tuple(logits.shape)}")
    h20 = entropy(logits, 20.0)
    h1 = entropy(logits, 1.0)
    log_p1 = torch.log_softmax(logits, dim=1)
    ce1 = -log_p1[torch.arange(len(labels)), labels]
    probabilities1 = log_p1.exp()
    predicted = logits.argmax(dim=1)
    score_arrays = {"entropy_t20": h20, "entropy_t1": h1, "true_ce_t1": ce1}
    records = []
    for index, ((path, target), label) in enumerate(zip(dataset.samples, labels.tolist())):
        if target != label:
            raise RuntimeError("dataset label order mismatch")
        source = Path(path).resolve()
        records.append(
            {
                "sample_id": source.relative_to(source_root).as_posix(),
                "source_path": str(source),
                "source_sha256": sha256(source),
                "class_id": label,
                "class_folder": dataset.classes[label],
                "entropy_t20": float(h20[index]),
                "entropy_t1": float(h1[index]),
                "true_ce_t1": float(ce1[index]),
                "teacher_true_probability_t1": float(probabilities1[index, label]),
                "teacher_prediction": int(predicted[index]),
                "teacher_correct": bool(predicted[index] == label),
            }
        )
    np.savez_compressed(
        args.output_root / "teacher_logits.npz",
        logits=logits.numpy().astype(np.float32), labels=labels.numpy().astype(np.int64),
        sample_ids=np.asarray([record["sample_id"] for record in records]),
    )
    rankings = {method: {} for method in METHODS}
    selections = {method: {} for method in METHODS}
    for class_id in range(100):
        class_records = [record for record in records if record["class_id"] == class_id]
        for method in METHODS:
            ranked = sorted(class_records, key=lambda row: (row[method], row["sample_id"]))
            rankings[method][str(class_id)] = [row["sample_id"] for row in ranked]
            for ipc in IPCS:
                selections[method].setdefault(str(ipc), []).extend(ranked[:ipc])
    for method in METHODS:
        for ipc in IPCS:
            selected = selections[method][str(ipc)]
            output = args.output_root / "selected" / method / f"ipc{ipc}"
            link_selection(selected, output)
    overlaps = {}
    pairs = (("entropy_t20", "entropy_t1"), ("entropy_t20", "true_ce_t1"), ("entropy_t1", "true_ce_t1"))
    for first, second in pairs:
        key = f"{first}__vs__{second}"
        overlaps[key] = {}
        for ipc in IPCS:
            per_class = []
            identical = 0
            for class_id in range(100):
                a = set(rankings[first][str(class_id)][:ipc])
                b = set(rankings[second][str(class_id)][:ipc])
                count = len(a & b)
                per_class.append(count)
                identical += int(a == b)
            overlaps[key][str(ipc)] = {
                "intersection": sum(per_class),
                "selected_per_method": 100 * ipc,
                "overlap_fraction": sum(per_class) / (100 * ipc),
                "classes_with_identical_sets": identical,
                "per_class_intersection": per_class,
            }
    selection_summaries = {}
    for method in METHODS:
        selection_summaries[method] = {}
        for ipc in IPCS:
            selected = selections[method][str(ipc)]
            selection_summaries[method][str(ipc)] = {
                "images": len(selected),
                "teacher_accuracy": sum(row["teacher_correct"] for row in selected) / len(selected),
                "mean_entropy_t20": sum(row["entropy_t20"] for row in selected) / len(selected),
                "mean_entropy_t1": sum(row["entropy_t1"] for row in selected) / len(selected),
                "mean_true_ce_t1": sum(row["true_ce_t1"] for row in selected) / len(selected),
                "mean_teacher_true_probability_t1": sum(row["teacher_true_probability_t1"] for row in selected) / len(selected),
            }
    payload = {
        "status": "complete",
        "method": "per-class deterministic Teacher score ranking",
        "dataset": "A_imsize224",
        "source_root": str(source_root),
        "images": len(dataset),
        "classes": len(dataset.classes),
        "class_names": dataset.classes,
        "teacher_checkpoint": str(args.teacher.resolve()),
        "teacher_sha256": sha256(args.teacher),
        "teacher_mode": "eval",
        "input_view": "existing complete physical 224x224 RGB image",
        "normalization": {"mean": list(MEAN), "std": list(STD)},
        "logits": str((args.output_root / "teacher_logits.npz").resolve()),
        "logits_dtype": "float32",
        "ranking": {"primary": "entropy_t20 ascending", "ties": "sample_id ascending", "natural_log": True},
        "ipcs": list(IPCS),
        "nested": True,
        "overlaps": overlaps,
        "selection_summaries": selection_summaries,
        "rankings": rankings,
        "records": records,
    }
    atomic_json(payload, manifest_path)
    print(json.dumps({"status": "complete", "manifest": str(manifest_path.resolve()), "overlaps": overlaps, "selection_summaries": selection_summaries}, indent=2))


if __name__ == "__main__":
    main()
