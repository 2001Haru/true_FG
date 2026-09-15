"""NRR-inspired constrained optimization of Aircraft RandomReal IPC3 pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models


MEAN = (0.4865, 0.5177, 0.5425)
STD = (0.2124, 0.2051, 0.2375)
ARMS = ("plain", "cam_protected", "random_protected")
SNAPSHOTS = (0, 100, 500, 1000, 2000)
PROTECTED_PIXELS = 15053


def stable_u64(*values):
    return int.from_bytes(hashlib.sha256("\0".join(map(str, values)).encode()).digest()[:8], "little")


def stable_nonzero_shift(seed, class_id, identity, size=224):
    counter = 0
    while True:
        digest = stable_u64("nrr-random-mask-shift-v1", seed, class_id, identity, counter)
        dy, dx = digest % size, (digest // size) % size
        if dy != 0 or dx != 0:
            return int(dy), int(dx), counter
        counter += 1


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_rgb(path):
    with Image.open(path) as handle:
        array = torch.from_numpy(__import__("numpy").array(handle.convert("RGB"), dtype="uint8").copy())
    if tuple(array.shape) != (224, 224, 3):
        raise RuntimeError(f"expected decoded 224x224 RGB image: {path}, got {tuple(array.shape)}")
    return array.permute(2, 0, 1).float().div_(255.0)


def save_rgb(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = tensor.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray(pixels, mode="RGB")
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG", compress_level=0)
    os.replace(temporary, path)


def load_mask(path):
    with Image.open(path) as handle:
        array = torch.from_numpy(__import__("numpy").array(handle.convert("L"), dtype="uint8").copy())
    return array.ne(0)


def save_mask(mask, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(mask.byte().mul(255).cpu().numpy(), mode="L")
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG", compress_level=0)
    os.replace(temporary, path)


def normalize(images):
    mean = images.new_tensor(MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(STD).view(1, 3, 1, 1)
    return (images - mean) / std


def load_teacher(path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "ResNet18" in state:
        state = state["ResNet18"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if sum(parameter.numel() for parameter in model.parameters()) != 11227812:
        raise RuntimeError("unexpected Teacher parameter count")
    return model


class BNInputHook:
    def __init__(self, module):
        self.running_mean = module.running_mean.detach().clone()
        self.running_var = module.running_var.detach().clone()
        self.value = None
        self.handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        feature = inputs[0]
        mean = feature.mean(dim=(0, 2, 3))
        variance = feature.var(dim=(0, 2, 3), correction=0)
        self.value = torch.linalg.vector_norm(mean - self.running_mean) + torch.linalg.vector_norm(variance - self.running_var)

    def close(self):
        self.handle.remove()


def cosine_lr(update, updates, initial, minimum):
    if updates <= 1:
        return minimum
    position = (update - 1) / (updates - 1)
    return minimum + (initial - minimum) * (1 + math.cos(math.pi * position)) / 2


def flip_flags(count, seed, slot, update):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_u64("nrr-construction-flip-v1", seed, slot, update))
    return torch.rand(count, generator=generator).lt(0.5)


def apply_flips(images, flags):
    return torch.where(flags.to(images.device)[:, None, None, None], images.flip(-1), images)


def topk_mask(values, count):
    flat = values.flatten()
    threshold = torch.topk(flat, count, largest=True, sorted=False).values.min()
    mask = flat.gt(threshold)
    remaining = count - int(mask.sum())
    if remaining:
        ties = flat.eq(threshold).nonzero(as_tuple=False).flatten()
        mask[ties[:remaining]] = True
    if int(mask.sum()) != count:
        raise RuntimeError("top-k mask area mismatch")
    return mask.view_as(values)


def project_pixels(images, initial, protected, epsilon):
    lower = (initial - epsilon).clamp_min(0)
    upper = (initial + epsilon).clamp_max(1)
    projected = torch.maximum(lower, torch.minimum(upper, images))
    return torch.where(protected[:, None], initial, projected)


def gradient_stats(tensor):
    flat = tensor.detach().float().reshape(-1)
    return {
        "l2": float(torch.linalg.vector_norm(flat)),
        "rms": float(flat.square().mean().sqrt()),
        "mean_abs": float(flat.abs().mean()),
        "max_abs": float(flat.abs().max()),
    }


def displacement_stats(images, reference, protected):
    delta = images.detach() - reference
    absolute = delta.abs()
    editable = (~protected)[:, None].expand_as(absolute)
    editable_absolute = absolute.masked_select(editable)
    return {
        "mean_abs_all": float(absolute.mean()),
        "rms_all": float(delta.square().mean().sqrt()),
        "max_abs_all": float(absolute.max()),
        "changed_channel_fraction_all": float(absolute.gt(0).float().mean()),
        "mean_abs_editable": float(editable_absolute.mean()),
        "rms_editable": float(editable_absolute.square().mean().sqrt()),
        "max_abs_editable": float(editable_absolute.max()),
        "protected_max_abs": float(absolute.masked_select(protected[:, None].expand_as(absolute)).max()) if protected.any() else 0.0,
    }


def cumulative_pixel_stats(images, initial, protected, epsilon):
    delta = images.detach() - initial
    absolute = delta.abs()
    editable = (~protected)[:, None].expand_as(absolute)
    base = displacement_stats(images, initial, protected)
    base.update({
        "linf_bound_fraction_editable": float(absolute.masked_select(editable).ge(epsilon - 1e-7).float().mean()),
        "rgb_boundary_fraction_editable": float(
            (images.detach().masked_select(editable).le(1e-7) | images.detach().masked_select(editable).ge(1 - 1e-7)).float().mean()
        ),
    })
    return base


def read_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if protocol.get("protocol") != "aircraft_ipc3_nrr_inspired_pixel_optimization_v1":
        raise RuntimeError("wrong protocol file")
    return protocol


def prepare(args):
    protocol = read_protocol(args.protocol)
    selection = json.loads(args.selection_manifest.read_text())
    if selection.get("status") != "complete" or selection.get("dataset") != "A_imsize224" or selection.get("ipc") != 3 or selection.get("selection_seed") != 0:
        raise RuntimeError("expected completed Aircraft RandomReal seed0 IPC3 manifest")
    rows = sorted(selection["images"], key=lambda row: (int(row["rank"]), int(row["class_id"])))
    if len(rows) != 300:
        raise RuntimeError("selection must contain 300 images")
    for slot in range(3):
        slot_rows = [row for row in rows if int(row["rank"]) == slot]
        if [int(row["class_id"]) for row in slot_rows] != list(range(100)):
            raise RuntimeError(f"slot {slot} is not one image per class")
    device = torch.device(args.device)
    teacher = load_teacher(args.teacher, device)
    activation = {}
    handle = teacher.layer4.register_forward_hook(lambda module, inputs, output: activation.__setitem__("layer4", output))
    output_rows, correct = [], 0
    overlaps, cam_ranges = [], []
    for slot in range(3):
        slot_rows = [row for row in rows if int(row["rank"]) == slot]
        images = torch.stack([load_rgb(row["selected_path"]) for row in slot_rows]).to(device)
        labels = torch.arange(100, device=device)
        with torch.no_grad():
            logits = teacher(normalize(images))
            features = activation["layer4"]
            cams = torch.einsum("bchw,bc->bhw", features, teacher.fc.weight[labels]).relu()
            cams = F.interpolate(cams[:, None], (224, 224), mode="bilinear", align_corners=False)[:, 0].cpu()
        correct += int(logits.argmax(1).eq(labels).sum())
        for row, cam in zip(slot_rows, cams):
            cam_mask = topk_mask(cam, PROTECTED_PIXELS)
            dy, dx, redraw_counter = stable_nonzero_shift(
                args.seed, row["class_id"], Path(row["selected_path"]).name
            )
            random_mask = torch.roll(cam_mask, shifts=(int(dy), int(dx)), dims=(0, 1))
            name = Path(row["selected_path"]).stem + ".png"
            cam_path = args.output_root / "masks/cam" / row["class_folder"] / name
            random_path = args.output_root / "masks/random" / row["class_folder"] / name
            save_mask(cam_mask, cam_path)
            save_mask(random_mask, random_path)
            overlap = float((cam_mask & random_mask).sum() / PROTECTED_PIXELS)
            overlaps.append(overlap)
            cam_ranges.append(float(cam.max() - cam.min()))
            output_rows.append({
                "class_id": int(row["class_id"]), "class_folder": row["class_folder"], "slot": int(row["rank"]),
                "identity": Path(row["selected_path"]).name, "selected_path": str(Path(row["selected_path"]).resolve()),
                "source_path": str(Path(row["source_path"]).resolve()), "source_sha256": row["source_sha256"],
                "cam_mask": str(cam_path.resolve()), "random_mask": str(random_path.resolve()),
                "random_shift_yx": [dy, dx], "random_shift_redraw_counter": redraw_counter,
                "mask_overlap_fraction": overlap,
                "cam_min": float(cam.min()), "cam_max": float(cam.max()),
            })
    handle.remove()
    initial_digest = hashlib.sha256()
    for row in output_rows:
        image = load_rgb(row["selected_path"])
        initial_digest.update(f"{row['class_folder']}/{row['identity']}".encode())
        initial_digest.update(image.mul(255).byte().numpy().tobytes())
        if load_mask(row["cam_mask"]).sum().item() != PROTECTED_PIXELS or load_mask(row["random_mask"]).sum().item() != PROTECTED_PIXELS:
            raise RuntimeError("protection mask area mismatch")
    result = {
        "status": "complete", "protocol": protocol["protocol"], "stage": "prepare", "seed": args.seed,
        "selection_manifest": str(args.selection_manifest.resolve()), "selection_manifest_sha256": sha256_file(args.selection_manifest),
        "teacher": str(args.teacher.resolve()), "teacher_sha256": sha256_file(args.teacher), "teacher_mode": "eval",
        "teacher_parameters": 11227812, "teacher_trainable_parameters": 0, "teacher_initial_top1": 100 * correct / 300,
        "decoded_initial_tree_sha256": initial_digest.hexdigest(), "images": 300, "classes": 100, "ipc": 3,
        "protected_pixels_per_image": PROTECTED_PIXELS, "protected_fraction": PROTECTED_PIXELS / (224 * 224),
        "random_mask_overlap_mean": sum(overlaps) / len(overlaps),
        "zero_range_cam_images": sum(value == 0 for value in cam_ranges), "rows": output_rows,
    }
    atomic_json(result, args.output_root / "initial_manifest.json")
    print(json.dumps({key: result[key] for key in ("status", "images", "teacher_initial_top1", "protected_fraction", "random_mask_overlap_mean", "zero_range_cam_images")}, indent=2))


def objective(teacher, hooks, images, labels, flags, bn_coefficient):
    logits = teacher(normalize(apply_flips(images, flags)))
    ce = F.cross_entropy(logits.float(), labels, reduction="mean")
    if any(hook.value is None for hook in hooks):
        raise RuntimeError("BN hook did not run")
    weighted_terms = [(10.0 if index == 0 else 1.0) * hook.value for index, hook in enumerate(hooks)]
    bn_raw = torch.stack(weighted_terms).sum()
    bn_weighted = bn_coefficient * bn_raw
    return ce, bn_raw, bn_weighted


def optimize(args):
    protocol = read_protocol(args.protocol)
    if args.arm not in ARMS:
        raise ValueError(args.arm)
    initial_manifest = json.loads((args.output_root / "initial_manifest.json").read_text())
    if initial_manifest.get("status") != "complete" or initial_manifest.get("teacher_sha256") != sha256_file(args.teacher):
        raise RuntimeError("invalid initial manifest or Teacher collision")
    device = torch.device(args.device)
    teacher = load_teacher(args.teacher, device)
    hooks = [BNInputHook(module) for module in teacher.modules() if isinstance(module, nn.BatchNorm2d)]
    if len(hooks) != 20:
        raise RuntimeError(f"expected 20 BatchNorm2d layers, found {len(hooks)}")
    arm_slots = []
    for slot in range(3):
        slot_manifest_path = args.output_root / f"arm_slots/{args.arm}/slot{slot}.json"
        if slot_manifest_path.is_file():
            cached = json.loads(slot_manifest_path.read_text())
            if cached.get("status") == "complete" and cached.get("updates") == args.updates:
                arm_slots.append(cached)
                print(f"reusing completed {args.arm} slot{slot}", flush=True)
                continue
        rows = sorted((row for row in initial_manifest["rows"] if row["slot"] == slot), key=lambda row: row["class_id"])
        if len(rows) != 100:
            raise RuntimeError(f"slot{slot} does not contain 100 images")
        initial = torch.stack([load_rgb(row["selected_path"]) for row in rows]).to(device)
        images = nn.Parameter(initial.clone())
        labels = torch.arange(100, device=device)
        if args.arm == "plain":
            protected = torch.zeros(100, 224, 224, dtype=torch.bool, device=device)
        else:
            key = "cam_mask" if args.arm == "cam_protected" else "random_mask"
            protected = torch.stack([load_mask(row[key]) for row in rows]).to(device)
        optimizer = torch.optim.Adam([images], lr=args.lr, betas=(0.5, 0.9), eps=1e-8, weight_decay=0)
        trajectory = []
        for row, tensor in zip(rows, initial):
            save_rgb(tensor, args.output_root / f"snapshots/{args.arm}/step0/{row['class_folder']}/{Path(row['identity']).stem}.png")
        for update in range(1, args.updates + 1):
            lr = cosine_lr(update, args.updates, args.lr, args.min_lr)
            optimizer.param_groups[0]["lr"] = lr
            flags = flip_flags(100, args.seed, slot, update)
            optimizer.zero_grad(set_to_none=True)
            ce, bn_raw, bn_weighted = objective(teacher, hooks, images, labels, flags, args.bn_coefficient)
            loss = ce + bn_weighted
            diagnostic = update in (1, 100, 500, 1000, args.updates)
            if diagnostic:
                previous_images = images.detach().clone()
                ce_gradient = torch.autograd.grad(ce, images, retain_graph=True)[0]
                bn_gradient = torch.autograd.grad(bn_weighted, images, retain_graph=True)[0]
                denominator = torch.linalg.vector_norm(ce_gradient) * torch.linalg.vector_norm(bn_gradient)
                cosine = float((ce_gradient * bn_gradient).sum() / denominator) if float(denominator) > 0 else 0.0
                if images.grad is not None:
                    raise RuntimeError("component gradient audit unexpectedly populated parameter .grad")
            loss.backward()
            if protected.any():
                images.grad.masked_fill_(protected[:, None], 0)
            if diagnostic:
                total_gradient = gradient_stats(images.grad)
            optimizer.step()
            with torch.no_grad():
                images.copy_(project_pixels(images, initial, protected, args.epsilon))
            if diagnostic:
                trajectory.append({
                    "update": update, "lr": lr, "ce": float(ce), "bn_raw": float(bn_raw),
                    "bn_weighted": float(bn_weighted), "loss": float(loss),
                    "ce_pixel_gradient": gradient_stats(ce_gradient),
                    "weighted_bn_pixel_gradient": gradient_stats(bn_gradient),
                    "ce_bn_gradient_cosine": cosine, "effective_total_gradient": total_gradient,
                    "single_step_displacement": displacement_stats(images, previous_images, protected),
                    "cumulative_displacement": cumulative_pixel_stats(images, initial, protected, args.epsilon),
                    "flipped_images": int(flags.sum()),
                })
            if update in SNAPSHOTS[1:]:
                for row, tensor in zip(rows, images.detach()):
                    save_rgb(tensor, args.output_root / f"snapshots/{args.arm}/step{update}/{row['class_folder']}/{Path(row['identity']).stem}.png")
        exported = []
        for row, tensor, original, mask in zip(rows, images.detach(), initial, protected):
            path = args.output_root / f"selected/{args.arm}/ipc3/{row['class_folder']}/{Path(row['identity']).stem}.png"
            save_rgb(tensor, path)
            decoded = load_rgb(path).to(device)
            delta = (decoded - original).abs()
            editable = (~mask)[None].expand_as(delta)
            editable_delta = delta.masked_select(editable)
            editable_pixels = decoded.masked_select(editable)
            if float(delta.max()) > args.epsilon + 0.5 / 255 + 1e-7:
                raise RuntimeError("quantized export exceeds pixel constraint")
            if mask.any() and float(delta.masked_select(mask[None].expand_as(delta)).max()) != 0:
                raise RuntimeError("protected export pixel changed")
            exported.append({
                "class_id": row["class_id"], "class_folder": row["class_folder"], "slot": slot,
                "identity": row["identity"], "path": str(path.resolve()), "sha256": sha256_file(path),
                "mean_abs_change": float(delta.mean()), "rms_change": float(delta.square().mean().sqrt()),
                "max_abs_change": float(delta.max()),
                "bound_hit_fraction_editable": float(editable_delta.ge(args.epsilon - 1e-7).float().mean()),
                "rgb_boundary_fraction_editable": float(
                    (editable_pixels.eq(0) | editable_pixels.eq(1)).float().mean()
                ),
                "protected_max_abs": float(delta.masked_select(mask[None].expand_as(delta)).max()) if mask.any() else 0.0,
            })
        slot_result = {
            "status": "complete", "arm": args.arm, "slot": slot, "updates": args.updates,
            "images": 100, "examples_optimized": 100 * args.updates, "trajectory": trajectory, "exports": exported,
        }
        atomic_json(slot_result, slot_manifest_path)
        arm_slots.append(slot_result)
    for hook in hooks:
        hook.close()
    exports = [row for slot in arm_slots for row in slot["exports"]]
    if len(exports) != 300 or len({(row["class_id"], row["slot"]) for row in exports}) != 300:
        raise RuntimeError("arm export is incomplete")
    result = {
        "status": "complete", "protocol": protocol["protocol"], "stage": "optimize", "arm": args.arm,
        "teacher_sha256": sha256_file(args.teacher), "initial_manifest_sha256": sha256_file(args.output_root / "initial_manifest.json"),
        "updates_per_image": args.updates, "batch_size": 100, "batch_updates": args.updates * 3,
        "image_forwards": args.updates * 300, "optimizer": "Adam", "lr": args.lr, "min_lr": args.min_lr,
        "betas": [0.5, 0.9], "eps": 1e-8, "bn_coefficient": args.bn_coefficient,
        "pixel_epsilon": args.epsilon, "snapshots": list(SNAPSHOTS), "slots": arm_slots, "exports": exports,
    }
    atomic_json(result, args.output_root / f"arm_manifests/{args.arm}.json")
    print(json.dumps({"status": "complete", "arm": args.arm, "exports": len(exports)}, indent=2))


def finalize(args):
    protocol = read_protocol(args.protocol)
    initial = json.loads((args.output_root / "initial_manifest.json").read_text())
    arms, summaries = {}, {}
    for arm in ARMS:
        path = args.output_root / f"arm_manifests/{arm}.json"
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete" or payload.get("arm") != arm or len(payload.get("exports", [])) != 300:
            raise RuntimeError(f"invalid arm manifest: {arm}")
        for row in payload["exports"]:
            if sha256_file(row["path"]) != row["sha256"]:
                raise RuntimeError(f"export collision: {row['path']}")
        changes = [row["mean_abs_change"] for row in payload["exports"]]
        rms = [row["rms_change"] for row in payload["exports"]]
        hits = [row["bound_hit_fraction_editable"] for row in payload["exports"]]
        boundaries = [row["rgb_boundary_fraction_editable"] for row in payload["exports"]]
        protected = [row["protected_max_abs"] for row in payload["exports"]]
        summaries[arm] = {
            "mean_image_mean_abs_change": sum(changes) / len(changes),
            "mean_image_rms_change": sum(rms) / len(rms),
            "mean_bound_hit_fraction_editable": sum(hits) / len(hits),
            "mean_rgb_boundary_fraction_editable": sum(boundaries) / len(boundaries),
            "protected_max_abs": max(protected),
        }
        arms[arm] = {"manifest": str(path.resolve()), "manifest_sha256": sha256_file(path),
                     "image_root": str((args.output_root / f"selected/{arm}/ipc3").resolve())}
    result = {
        "status": "complete", "protocol": protocol["protocol"], "stage": "construction_complete",
        "initial_manifest": str((args.output_root / "initial_manifest.json").resolve()),
        "initial_manifest_sha256": sha256_file(args.output_root / "initial_manifest.json"),
        "teacher_sha256": initial["teacher_sha256"], "images_per_arm": 300, "arms": arms,
        "pixel_audit": summaries, "new_fkd_sets": 3, "new_student_runs": 9,
    }
    atomic_json(result, args.output_root / "construction_manifest.json")
    print(json.dumps(result, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "optimize", "finalize"))
    p.add_argument("--protocol", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--teacher", type=Path)
    p.add_argument("--selection-manifest", type=Path)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--updates", default=2000, type=int)
    p.add_argument("--lr", default=0.01, type=float)
    p.add_argument("--min-lr", default=1e-4, type=float)
    p.add_argument("--bn-coefficient", default=0.01, type=float)
    p.add_argument("--epsilon", default=32 / 255, type=float)
    args = p.parse_args()
    if args.stage in ("prepare", "optimize") and args.teacher is None:
        p.error("--teacher is required")
    if args.stage == "prepare" and args.selection_manifest is None:
        p.error("--selection-manifest is required")
    if args.stage == "optimize" and args.arm is None:
        p.error("--arm is required")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    {"prepare": prepare, "optimize": optimize, "finalize": finalize}[arguments.stage](arguments)
