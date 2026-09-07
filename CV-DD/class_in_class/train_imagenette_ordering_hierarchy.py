"""Train one O/B/A/Aprime ordering-hierarchy Student with frozen paired semantics."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from imagenette_entropy_protocol import (
    BATCH_SIZE, CLASSES, DeterministicReleasedView, IMAGE_SIZE, IPC, LR,
    MOMENTUM, TEST_IMAGES, TOTAL_UPDATES, TRAIN_EPOCHS, TRAIN_SIZE,
    WEIGHT_DECAY, atomic_json, build_student, build_teacher, file_sha256,
    identity_seed, lr_for_epoch, normalize, state_dict_sha256, test_transform,
)
from prepare_imagenette_ordering_hierarchy import apply_mapping, mapping_for_order
from train_imagenette_entropy_selection import validate


EVAL_EPOCHS = (200, 400, 600, 800, 1000, 1200, 1333, 1400, 1600, 1666, 1800, 2000)


class OrderingDataset(Dataset):
    def __init__(self, manifest_path):
        self.manifest_path = manifest_path.resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.rows = sorted(self.manifest["images"], key=lambda row: (int(row["class_id"]), row["relative_path"]))
        if len(self.rows) != TRAIN_SIZE or len({row["relative_path"] for row in self.rows}) != TRAIN_SIZE:
            raise RuntimeError("invalid IPC10 ordering manifest")
        high = set()
        for class_id in range(CLASSES):
            rows = [row for row in self.rows if int(row["class_id"]) == class_id]
            ordered = sorted(rows, key=lambda row: (float(row["calibration_entropy"]), row["relative_path"]))
            if len(ordered) != IPC:
                raise RuntimeError("manifest is not class-balanced IPC10")
            high.update(row["relative_path"] for row in ordered[5:])
        self.high = high
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.transform = DeterministicReleasedView()

    def bind_student(self, seed):
        self.student_seed = seed

    def set_epoch(self, epoch):
        self.epoch.fill_(epoch)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        seed = identity_seed(
            "imagenette-entropy-student-view-v1", self.student_seed,
            int(self.epoch.item()), row["relative_path"],
        )
        with Image.open(row["source_path"]) as image:
            tensor = self.transform(image, seed)
        return tensor, int(row["class_id"]), row["relative_path"] in self.high, index


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reordered_targets(probabilities, targets, row_indices, dataset, templates, arm):
    if arm == "O":
        return probabilities, [list(range(CLASSES)) for _ in range(len(targets))]
    output = []
    mappings = []
    probabilities_cpu = probabilities.cpu()
    targets_cpu = targets.cpu()
    selection_seed = str(dataset.manifest["selection_seed"])
    for probability, target_tensor, row_index_tensor in zip(probabilities_cpu, targets_cpu, row_indices):
        target = int(target_tensor)
        row = dataset.rows[int(row_index_tensor)]
        if arm == "B":
            destination = templates["image_templates"][row["relative_path"]]["full_order"]
        else:
            destination = templates["class_templates"][selection_seed][str(target)]["order"]
        mapping = mapping_for_order(probability.cpu(), target, destination)
        if arm == "Aprime":
            mapping = np.argsort(mapping).tolist()
        output.append(apply_mapping(probability.cpu(), mapping))
        mappings.append(mapping)
    return torch.stack(output).to(probabilities.device), mappings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--templates", required=True, type=Path)
    parser.add_argument("--test-dir", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--protocol-spec", required=True, type=Path)
    parser.add_argument("--arm", choices=("O", "B", "A", "Aprime"), required=True)
    parser.add_argument("--student-seed", choices=(42, 43, 44), required=True, type=int)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--workers", default=4, type=int)
    args = parser.parse_args()
    seed_everything(identity_seed("imagenette-training-global-v1", args.student_seed))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(identity_seed("imagenette-student-init-v1", args.student_seed))
        student = build_student()
    initial_hash = state_dict_sha256(student.state_dict())
    dataset = OrderingDataset(args.manifest)
    dataset.bind_student(args.student_seed)
    templates = json.loads(args.templates.read_text(encoding="utf-8"))
    manifest_record = templates["manifests"][str(dataset.manifest["selection_seed"])]
    if manifest_record["sha256"] != file_sha256(args.manifest.resolve()):
        raise RuntimeError("template file does not bind this manifest")
    test_dataset = datasets.ImageFolder(args.test_dir.resolve(), transform=test_transform)
    if len(test_dataset) != TEST_IMAGES:
        raise RuntimeError("not the official ImageNette test split")
    generator = torch.Generator().manual_seed(identity_seed("imagenette-shuffle-v1", args.student_seed))
    train_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator,
        num_workers=args.workers, persistent_workers=args.workers > 0,
        pin_memory=True, prefetch_factor=2 if args.workers > 0 else None,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=args.workers,
        persistent_workers=args.workers > 0, pin_memory=True,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    device = torch.device("cuda")
    student = student.to(device)
    teacher = build_teacher(args.teacher_checkpoint.resolve()).to(device).eval()
    optimizer = torch.optim.SGD(student.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    history = []
    updates = 0
    best_top1 = float("-inf")
    best_epoch = None
    final_metrics = None
    teacher_sums = {"entropy": 0.0, "true_probability": 0.0, "maximum_probability": 0.0, "correct": 0.0}
    teacher_views = 0
    invariant_max = defaultdict_float = {
        "true_probability": 0.0, "non_target_mass": 0.0, "sorted_multiset": 0.0,
        "entropy": 0.0, "k2": 0.0, "el2n": 0.0, "maximum_probability": 0.0,
        "top2_total_mass": 0.0,
    }
    correctness_mismatches = 0
    changed_incorrect_argmax = 0
    intervention_l2_sum = 0.0
    intervention_views = 0
    started = time.time()
    for epoch in range(1, TRAIN_EPOCHS + 1):
        dataset.set_epoch(epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr_for_epoch(epoch)
        student.train()
        epoch_loss_sum = 0.0
        epoch_examples = 0
        batch_sizes = []
        for raw, targets, high, row_indices in train_loader:
            raw = raw.to(device, non_blocking=True); targets = targets.to(device, non_blocking=True)
            high_cpu = high.bool()
            high = high_cpu.to(device, non_blocking=True); images = normalize(raw)
            optimizer.zero_grad(set_to_none=True)
            logits = student(images)
            with torch.no_grad():
                original = F.softmax(teacher(images[high]).float(), dim=1)
                changed, _ = reordered_targets(original, targets[high], row_indices[high_cpu], dataset, templates, args.arm)
                true_original = original.gather(1, targets[high, None]).squeeze(1)
                true_changed = changed.gather(1, targets[high, None]).squeeze(1)
                original_entropy = -(original * original.clamp_min(1e-30).log()).sum(1)
                changed_entropy = -(changed * changed.clamp_min(1e-30).log()).sum(1)
                onehot = F.one_hot(targets[high], num_classes=CLASSES).float()
                original_non_target_max = (original - onehot).max(1).values
                changed_non_target_max = (changed - onehot).max(1).values
                values = {
                    "true_probability": (true_changed - true_original).abs().max(),
                    "non_target_mass": ((1-true_changed)-(1-true_original)).abs().max(),
                    "sorted_multiset": (changed.sort(1).values-original.sort(1).values).abs().max(),
                    "entropy": (changed_entropy-original_entropy).abs().max(),
                    "k2": (1/changed.square().sum(1)-1/original.square().sum(1)).abs().max(),
                    "el2n": (torch.linalg.vector_norm(changed-onehot,dim=1)-torch.linalg.vector_norm(original-onehot,dim=1)).abs().max(),
                    "maximum_probability": (changed.max(1).values-original.max(1).values).abs().max(),
                    "top2_total_mass": ((true_changed+changed_non_target_max)-(true_original+original_non_target_max)).abs().max(),
                }
                for key, value in values.items(): invariant_max[key] = max(invariant_max[key], float(value))
                original_correct = original.argmax(1).eq(targets[high]); changed_correct = changed.argmax(1).eq(targets[high])
                correctness_mismatches += int((original_correct != changed_correct).sum())
                changed_incorrect_argmax += int(((~original_correct) & (original.argmax(1) != changed.argmax(1))).sum())
                intervention_l2_sum += float(torch.linalg.vector_norm(changed-original,dim=1).sum())
                intervention_views += len(original)
                confidence, prediction = original.max(1)
                teacher_sums["entropy"] += float(original_entropy.sum())
                teacher_sums["true_probability"] += float(true_original.sum())
                teacher_sums["maximum_probability"] += float(confidence.sum())
                teacher_sums["correct"] += float(prediction.eq(targets[high]).sum())
                teacher_views += len(original)
            logp = F.log_softmax(logits, dim=1)
            soft_sum = -(changed * logp[high]).sum()
            hard_sum = F.cross_entropy(logits[~high], targets[~high], reduction="sum")
            loss = (soft_sum + hard_sum) / len(targets)
            loss.backward(); optimizer.step(); updates += 1
            epoch_loss_sum += float(loss) * len(targets); epoch_examples += len(targets); batch_sizes.append(len(targets))
        if batch_sizes != [64, 36] or epoch_examples != 100 or updates != 2 * epoch:
            raise RuntimeError("training cardinality changed")
        if epoch in EVAL_EPOCHS:
            metrics = validate(student, test_loader, device)
            if epoch == TRAIN_EPOCHS: final_metrics = metrics
            record = {"epoch": epoch, "update": updates, "lr": lr_for_epoch(epoch), "train_loss": epoch_loss_sum/epoch_examples, "test_loss": metrics["loss"], "test_top1": metrics["top1"]}
            history.append(record)
            if metrics["top1"] > best_top1: best_top1 = metrics["top1"]; best_epoch = epoch
            print(json.dumps(record, sort_keys=True), flush=True)
    if final_metrics is None or updates != TOTAL_UPDATES:
        raise RuntimeError("training incomplete")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir / "final.pth.tar"
    torch.save({"protocol":"imagenette_ordering_hierarchy_v1","arm":args.arm,"student_seed":args.student_seed,"epoch":TRAIN_EPOCHS,"update":updates,"state_dict":student.state_dict(),"optimizer":optimizer.state_dict()},checkpoint)
    source = Path(__file__).resolve().parents[2] / "CoDA/test/resnet_ap.py"
    payload = {
        "status":"complete","protocol":"imagenette_ordering_hierarchy_v1","arm":args.arm,
        "protocol_spec":str(args.protocol_spec.resolve()),"protocol_spec_sha256":file_sha256(args.protocol_spec.resolve()),
        "dataset":"imagenet-nette","classes":CLASSES,"ipc":IPC,"train_images":100,"test_images":TEST_IMAGES,
        "selection_seed":dataset.manifest["selection_seed"],"student_seed":args.student_seed,
        "student_model":"CoDA_released_ResNetAP10","student_model_source":str(source.resolve()),"student_model_source_sha256":file_sha256(source),
        "student_initialization":"random","initial_student_state_sha256":initial_hash,"normalization_layer":"GroupNorm(C,C)_released_instance_semantics",
        "image_size":IMAGE_SIZE,"optimizer":"SGD","initial_lr":LR,"momentum":MOMENTUM,"weight_decay":WEIGHT_DECAY,
        "batch_size":BATCH_SIZE,"actual_batch_sizes_per_epoch":[64,36],"gradient_accumulation_steps":1,"drop_last":False,
        "epochs":TRAIN_EPOCHS,"updates_completed":updates,"scheduler":"released_multistep_epoch","lr_milestones_after_epochs":[1333,1666],"lr_gamma":0.2,
        "cutmix":False,"mixup":False,"evaluation_epochs":list(EVAL_EPOCHS),"evaluation_count":len(EVAL_EPOCHS),
        "teacher_temperature":1.0,"student_temperature":1.0,"teacher_mode":"eval_frozen_no_grad","teacher_and_student_view_pixels_identical":True,
        "high_entropy_images_per_epoch":50,"low_entropy_hard_images_per_epoch":50,"loss":"batchmean_mixed_high_entropy_ordered_soft1_low_entropy_hard_ce",
        "primary_metric":"final_true_label_nll_at_update_4000","final_loss":final_metrics["loss"],"final_top1":final_metrics["top1"],
        "best_top1_diagnostic":best_top1,"best_epoch_diagnostic":best_epoch,"test_history":history,"per_class_final":final_metrics["per_class"],
        "teacher_online_original_statistics":{"views":teacher_views,**{f"mean_{key}":value/teacher_views for key,value in teacher_sums.items()}},
        "label_invariant_audit":{"views":intervention_views,"maximum_absolute_differences":invariant_max,"teacher_correctness_mismatches":correctness_mismatches,"incorrect_argmax_identity_changes":changed_incorrect_argmax,"mean_O_to_arm_l2":intervention_l2_sum/intervention_views},
        "train_manifest":str(args.manifest.resolve()),"train_manifest_sha256":file_sha256(args.manifest.resolve()),
        "ordering_templates":str(args.templates.resolve()),"ordering_templates_sha256":file_sha256(args.templates.resolve()),
        "teacher_checkpoint":str(args.teacher_checkpoint.resolve()),"teacher_checkpoint_sha256":file_sha256(args.teacher_checkpoint.resolve()),
        "final_checkpoint":str(checkpoint.resolve()),"final_checkpoint_sha256":file_sha256(checkpoint),"workers":args.workers,"persistent_workers":args.workers>0,"elapsed_seconds":time.time()-started,
    }
    atomic_json(args.result, payload)
    print(json.dumps({"status":"complete","result":str(args.result.resolve()),"final_loss":payload["final_loss"],"final_top1":payload["final_top1"]}))


if __name__ == "__main__":
    main()
