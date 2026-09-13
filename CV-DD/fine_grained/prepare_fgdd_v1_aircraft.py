"""Prepare FGDD-v1 Aircraft IPC3 selections from a fixed RandomReal seed."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


MEAN = (0.4865, 0.5177, 0.5425)
STD = (0.2124, 0.2051, 0.2375)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".ppm", ".pgm", ".tif", ".tiff", ".webp"}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def stable_order(paths, namespace):
    return sorted(paths, key=lambda path: hashlib.sha256(
        f"{namespace}\0{path.as_posix()}".encode()).digest())


def image_tensor(path, flipped):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = TF.resize(image, [224, 224], interpolation=InterpolationMode.BILINEAR)
        if flipped:
            image = TF.hflip(image)
        return TF.normalize(TF.to_tensor(image), MEAN, STD)


class ViewDataset(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, index):
        image_index, path, view = self.items[index]
        return image_tensor(path, bool(view)), image_index, view


def round_robin(class_indices):
    result = []
    maximum = max(map(len, class_indices))
    for rank in range(maximum):
        for indices in class_indices:
            if rank < len(indices): result.append(indices[rank])
    return result


def extract_features(paths, device, workers):
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = nn.Identity(); backbone.eval().to(device)
    for parameter in backbone.parameters(): parameter.requires_grad_(False)
    items = [(index, path, view) for index, path in enumerate(paths) for view in (0, 1)]
    loader = DataLoader(ViewDataset(items), batch_size=128, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0)
    features = torch.empty(len(paths), 2, 512, dtype=torch.float32)
    with torch.no_grad():
        for images, indices, views in loader:
            output = backbone(images.to(device, non_blocking=True)).float().cpu()
            features[indices.long(), views.long()] = output
    return features


def teacher_logits(paths, class_indices, teacher_path, device, workers):
    model = models.resnet18(weights=None); model.fc = nn.Linear(model.fc.in_features, 100)
    model.load_state_dict(torch.load(teacher_path, map_location="cpu", weights_only=False), strict=True)
    model.train().to(device)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    interleaved = round_robin(class_indices)
    items = [(index, paths[index], view) for view in (0, 1) for index in interleaved]
    order_digest = hashlib.sha256("\n".join(
        f"{view}\t{paths[index].as_posix()}" for index, _, view in items).encode()).hexdigest()
    loader = DataLoader(ViewDataset(items), batch_size=20, shuffle=False, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0)
    logits = torch.full((len(paths), 2, 100), float("nan"), dtype=torch.float32)
    batches = 0
    with torch.no_grad():
        for images, indices, views in loader:
            images = images.to(device, non_blocking=True)
            split = images.shape[0] // 2
            output = torch.cat((model(images[:split]), model(images[split:])), dim=0).float().cpu()
            logits[indices.long(), views.long()] = output
            batches += 1
    return logits, {"batch_size": 20, "forward_split": "floor(n/2)+remainder",
                    "ordered_items_sha256": order_digest, "batches": batches,
                    "views": len(items), "teacher_mode": "train", "mix": None}


def train_head(features, targets, anchor_indices, seed):
    random.seed(seed); torch.manual_seed(seed)
    head = nn.Linear(features.shape[-1], 100)
    initial_hash = state_sha256(head.state_dict())
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    anchor_tensor = torch.tensor(anchor_indices, dtype=torch.long)
    visits = [0] * len(anchor_indices)
    losses = []
    head.train()
    update = 0
    for pass_index in range(100):
        permutation = torch.randperm(len(anchor_indices), generator=generator)
        for offset in (0, 100):
            local = permutation[offset:offset + 100]
            indices = anchor_tensor[local]
            views = torch.randint(0, 2, (len(indices),), generator=generator)
            batch_features = features[indices, views]
            batch_targets = targets[indices]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(batch_features), batch_targets)
            loss.backward(); optimizer.step(); losses.append(float(loss)); update += 1
            for value in local.tolist(): visits[value] += 1
    if update != 200 or min(visits) != 100 or max(visits) != 100:
        raise RuntimeError("scorer update/visit budget mismatch")
    head.eval()
    with torch.no_grad():
        anchor_logits = head(features[anchor_tensor].reshape(-1, features.shape[-1]))
        anchor_targets = targets[anchor_tensor].repeat_interleave(2)
        accuracy = float(anchor_logits.argmax(1).eq(anchor_targets).float().mean())
    return head, {"seed": seed, "optimizer": "AdamW", "lr": 1e-3, "weight_decay": 1e-4,
                  "batch_size": 100, "updates": update, "passes_over_200_anchors": 100,
                  "views": "one deterministic RNG-selected original/flip view per visit",
                  "initial_head_sha256": initial_hash, "final_head_sha256": state_sha256(head.state_dict()),
                  "first_loss": losses[0], "last_loss": losses[-1], "anchor_two_view_accuracy": accuracy}


def class_gradient(features, probabilities, class_id):
    delta = probabilities.clone(); delta[..., class_id] -= 1.0
    weight = torch.einsum("nvk,nvh->kh", delta, features) / (features.shape[0] * 2)
    bias = delta.mean(dim=(0, 1))
    return weight, bias


def image_gradients(features, probabilities, class_id):
    delta = probabilities.clone(); delta[..., class_id] -= 1.0
    weight = torch.einsum("nvk,nvh->nkh", delta, features) / 2.0
    bias = delta.mean(dim=1)
    return weight, bias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--random-ipc3", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--scorer-seed", type=int, default=20260914)
    parser.add_argument("--anchor-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4); torch.manual_seed(args.scorer_seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    args.output_root.mkdir(parents=True, exist_ok=True); cache = args.output_root / "cache"; cache.mkdir(exist_ok=True)

    class_dirs = sorted(path for path in args.train_root.iterdir() if path.is_dir())
    if len(class_dirs) != 100: raise RuntimeError(f"expected 100 classes, found {len(class_dirs)}")
    paths, targets, class_all = [], [], []
    for class_id, directory in enumerate(class_dirs):
        members = sorted(path.resolve() for path in directory.iterdir()
                         if path.is_file() and path.suffix.lower() in EXTENSIONS)
        indices = list(range(len(paths), len(paths) + len(members)))
        paths.extend(members); targets.extend([class_id] * len(members)); class_all.append(indices)
    targets = torch.tensor(targets, dtype=torch.long)
    index_by_resolved = {path: index for index, path in enumerate(paths)}

    anchors, thirds, anchor_by_class = [], [], []
    for class_id in range(100):
        selected_dir = args.random_ipc3 / f"{class_id:03d}"
        selected = sorted(path.resolve() for path in selected_dir.iterdir()
                          if path.is_file() or path.is_symlink())
        if len(selected) != 3 or any(path not in index_by_resolved for path in selected):
            raise RuntimeError(f"invalid Random IPC3 class {class_id}")
        ordered = stable_order(selected, f"fgdd-v1-anchor-order\0{args.anchor_seed}\0{class_id}")
        current_anchors = [index_by_resolved[path] for path in ordered[:2]]
        anchors.extend(current_anchors); anchor_by_class.append(set(current_anchors))
        thirds.append(index_by_resolved[ordered[2]])
    anchor_set = set(anchors)
    candidates_by_class = [[index for index in indices if index not in anchor_by_class[c]]
                           for c, indices in enumerate(class_all)]
    candidate_indices = [index for indices in candidates_by_class for index in indices]

    features = extract_features(paths, args.device, args.workers)
    torch.save({"paths": [str(path) for path in paths], "features": features}, cache / "imagenet_features.pt")
    head, scorer = train_head(features, targets, anchors, args.scorer_seed)
    torch.save(head.state_dict(), cache / "scorer_head.pth")
    with torch.no_grad(): all_scorer_logits = head(features.reshape(-1, 512)).reshape(len(paths), 2, 100)
    scorer_probabilities = all_scorer_logits.softmax(dim=-1)

    teacher, teacher_context = teacher_logits(paths, candidates_by_class, args.teacher,
                                               args.device, args.workers)
    torch.save({"paths": [str(path) for path in paths], "logits": teacher}, cache / "teacher_candidate_logits.pt")

    class_means = torch.zeros(100, 100)
    class_grad_w = torch.zeros(100, 100, 512); class_grad_b = torch.zeros(100, 100)
    for class_id, indices in enumerate(candidates_by_class):
        class_means[class_id] = scorer_probabilities[indices].mean(dim=(0, 1))
        class_grad_w[class_id], class_grad_b[class_id] = class_gradient(
            features[indices], scorer_probabilities[indices], class_id)
    confusion = (class_means + class_means.T) / 2.0
    confusion.fill_diagonal_(-1.0)
    neighbors = confusion.topk(3, dim=1).indices

    r1, r2, r3 = [], [], []
    confidence_all = torch.full((len(paths),), float("nan"))
    score2_all = torch.full((len(paths),), float("nan"))
    score3_all = torch.full((len(paths),), float("nan"))
    details = []
    for class_id, indices_list in enumerate(candidates_by_class):
        indices = torch.tensor(indices_list, dtype=torch.long)
        f = features[indices]; ps_t1 = scorer_probabilities[indices]
        ce_w, ce_b = image_gradients(f, ps_t1, class_id)
        n = len(indices_list)
        own_w = (class_grad_w[class_id] * n - ce_w) / (n - 1)
        own_b = (class_grad_b[class_id] * n - ce_b) / (n - 1)
        other_mask = [value for value in range(100) if value != class_id]
        other_w = class_grad_w[other_mask].mean(0); other_b = class_grad_b[other_mask].mean(0)
        neighbor_ids = neighbors[class_id].tolist()
        neighbor_w = class_grad_w[neighbor_ids].mean(0); neighbor_b = class_grad_b[neighbor_ids].mean(0)
        ref2_w = 0.5 * own_w + 0.5 * other_w; ref2_b = 0.5 * own_b + 0.5 * other_b
        ref3_w = 0.5 * own_w + 0.25 * other_w + 0.25 * neighbor_w
        ref3_b = 0.5 * own_b + 0.25 * other_b + 0.25 * neighbor_b

        student_t20 = (all_scorer_logits[indices] / 20.0).softmax(dim=-1)
        teacher_t20 = (teacher[indices] / 20.0).softmax(dim=-1)
        soft_delta = (student_t20 - teacher_t20) / 20.0
        soft_w = torch.einsum("nvk,nvh->nkh", soft_delta, f) / 2.0
        soft_b = soft_delta.mean(dim=1)
        score2 = torch.einsum("nkh,nkh->n", ref2_w, soft_w) + torch.einsum("nk,nk->n", ref2_b, soft_b)
        score3 = torch.einsum("nkh,nkh->n", ref3_w, soft_w) + torch.einsum("nk,nk->n", ref3_b, soft_b)
        confidence = teacher[indices].log_softmax(dim=-1)[:, :, class_id].mean(dim=1)
        confidence_all[indices] = confidence; score2_all[indices] = score2; score3_all[indices] = score3
        tie_paths = [paths[index].as_posix() for index in indices_list]
        def choose(values):
            return max(range(n), key=lambda j: (float(values[j]), tuple(-b for b in tie_paths[j].encode())))
        j1, j2, j3 = choose(confidence), choose(score2), choose(score3)
        r1.append(indices_list[j1]); r2.append(indices_list[j2]); r3.append(indices_list[j3])
        details.append({
            "class_id": class_id, "anchors": [str(paths[index]) for index in sorted(anchor_by_class[class_id])],
            "r0_third": str(paths[thirds[class_id]]), "candidate_count": n,
            "r0_scores": {
                "mean_t1_true_log_probability": float(confidence[indices_list.index(thirds[class_id])]),
                "ordinary_gradient_score": float(score2[indices_list.index(thirds[class_id])]),
                "fg_gradient_score": float(score3[indices_list.index(thirds[class_id])]),
            },
            "neighbors": neighbor_ids,
            "neighbor_bidirectional_scores": [float(confusion[class_id, value]) for value in neighbor_ids],
            "r1": {"path": str(paths[indices_list[j1]]), "mean_t1_true_log_probability": float(confidence[j1])},
            "r2": {"path": str(paths[indices_list[j2]]), "gradient_score": float(score2[j2])},
            "r3": {"path": str(paths[indices_list[j3]]), "gradient_score": float(score3[j3])},
        })

    torch.save({"paths": [str(path) for path in paths], "confidence": confidence_all,
                "ordinary_gradient_score": score2_all, "fg_gradient_score": score3_all},
               cache / "candidate_scores.pt")

    selections = {"r0_random": thirds, "r1_teacher_confidence": r1,
                  "r2_ordinary_gain": r2, "r3_fg_confusion_gain": r3}
    for method, third_indices in selections.items():
        root = args.output_root / "selected" / method / "ipc3"
        for class_id in range(100):
            output_class = root / f"{class_id:03d}"; output_class.mkdir(parents=True, exist_ok=True)
            chosen = list(anchor_by_class[class_id]) + [third_indices[class_id]]
            if len(set(chosen)) != 3: raise RuntimeError(f"duplicate selection in {method}, class {class_id}")
            for index in sorted(chosen):
                destination = output_class / paths[index].name
                if destination.exists() or destination.is_symlink():
                    if destination.resolve() != paths[index]: raise RuntimeError(f"collision: {destination}")
                else: os.symlink(paths[index], destination)

    r0_existing = sorted(path.resolve() for path in args.random_ipc3.glob("*/*"))
    r0_exported = sorted(path.resolve() for path in (args.output_root / "selected/r0_random/ipc3").glob("*/*"))
    if r0_existing != r0_exported: raise RuntimeError("R0 export differs from existing Random IPC3")
    overlap = {}
    for left in selections:
        for right in selections:
            if left < right:
                overlap[f"{left}__{right}"] = sum(a == b for a, b in zip(selections[left], selections[right]))
    manifest = {
        "status": "complete", "dataset": "A_imsize224", "ipc": 3,
        "train_root": str(args.train_root.resolve()), "random_ipc3": str(args.random_ipc3.resolve()),
        "teacher": str(args.teacher.resolve()), "teacher_sha256": sha256_file(args.teacher),
        "normalization": {"mean": MEAN, "std": STD},
        "stored_view": "RGB bilinear Resize224; original and deterministic horizontal flip",
        "anchors": {"seed": args.anchor_seed, "rule": "SHA-256 order within existing Random seed0 triplet; first two",
                    "count": len(anchors)},
        "candidate_pool": {"images": len(candidate_indices), "excludes": "two anchors/class",
                           "includes_original_random_third": True},
        "scorer": scorer, "teacher_scoring_context": teacher_context,
        "neighbor_rule": "0.5*(mean p(k)|y=c + mean p(c)|y=k), Top3",
        "reference_views": "original and horizontal flip, equal weight",
        "reference_weights": {"ordinary": {"own": 0.5, "other99": 0.5, "neighbors": 0.0},
                              "fg": {"own": 0.5, "other99": 0.25, "neighbors": 0.25}},
        "candidate_loss": "T20 KL gradient, no T^2; original+flip equal weight; float32",
        "leave_one_out": "candidate image and both views removed from own-class reference mean",
        "selection_counts": {method: 300 for method in selections}, "third_choice_overlap_classes": overlap,
        "teacher_scoring_queries": teacher_context["views"], "feature_forward_views": len(paths) * 2,
        "details": details,
    }
    output = args.output_root / "selection_manifest.json"; temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n"); os.replace(temporary, output)
    print(json.dumps({"status": "complete", "candidate_images": len(candidate_indices),
                      "teacher_queries": teacher_context["views"], "overlap": overlap,
                      "scorer": scorer}, indent=2))


if __name__ == "__main__":
    main()
