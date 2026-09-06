"""Train paired Hard/Soft ImageNette IPC10 ResNetAP-10 students."""

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
    BATCH_SIZE,
    CLASSES,
    DeterministicReleasedView,
    EVAL_EVERY_EPOCHS,
    IMAGE_SIZE,
    IPC,
    LR,
    MOMENTUM,
    STUDENT_SEEDS,
    TEST_IMAGES,
    TOTAL_UPDATES,
    TRAIN_EPOCHS,
    TRAIN_SIZE,
    WEIGHT_DECAY,
    atomic_json,
    build_student,
    build_teacher,
    file_sha256,
    identity_seed,
    lr_for_epoch,
    normalize,
    state_dict_sha256,
    test_transform,
)


class PairedManifestDataset(Dataset):
    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path.resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if (
            self.manifest.get("status") != "complete"
            or self.manifest.get("selection_images") != TRAIN_SIZE
            or self.manifest.get("ipc") != IPC
        ):
            raise RuntimeError(f"invalid ImageNette entropy selection manifest: {manifest_path}")
        rows = sorted(
            self.manifest["images"],
            key=lambda row: (int(row["class_id"]), row["relative_path"]),
        )
        if len({row["relative_path"] for row in rows}) != TRAIN_SIZE:
            raise RuntimeError("selection manifest has duplicate images")
        counts = np.bincount([int(row["class_id"]) for row in rows], minlength=CLASSES)
        if not np.array_equal(counts, np.full(CLASSES, IPC)):
            raise RuntimeError("selection manifest is not class-balanced IPC10")
        high_entropy_paths = set()
        for class_id in range(CLASSES):
            class_rows = [row for row in rows if int(row["class_id"]) == class_id]
            if any("calibration_entropy" not in row for row in class_rows):
                raise RuntimeError("manifest lacks frozen per-image calibration entropy")
            ordered = sorted(
                class_rows,
                key=lambda row: (
                    float(row["calibration_entropy"]),
                    row["relative_path"],
                ),
            )
            high_entropy_paths.update(row["relative_path"] for row in ordered[IPC // 2 :])
        if len(high_entropy_paths) != TRAIN_SIZE // 2:
            raise RuntimeError("expected exactly five high-entropy images per class")
        self.high_entropy_paths = high_entropy_paths
        self.rows = rows
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.view = DeterministicReleasedView()

    def set_epoch(self, epoch: int):
        self.epoch.fill_(epoch)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        epoch = int(self.epoch.item())
        seed = identity_seed(
            "imagenette-entropy-student-view-v1",
            self.student_seed,
            epoch,
            row["relative_path"],
        )
        with Image.open(row["source_path"]) as image:
            tensor = self.view(image, seed)
        return (
            tensor,
            int(row["class_id"]),
            row["relative_path"] in self.high_entropy_paths,
        )

    def bind_student_seed(self, student_seed: int):
        self.student_seed = student_seed


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixed_hard_soft1_loss(student_logits, true_targets, soft_mask, teacher_probabilities):
    """Actual-batch mean with hard CE and Soft-1 CE assigned per sample."""
    if soft_mask.dtype != torch.bool or soft_mask.shape != true_targets.shape:
        raise RuntimeError("invalid per-sample Soft-1 allocation mask")
    if teacher_probabilities.shape != (int(soft_mask.sum()), student_logits.shape[1]):
        raise RuntimeError("Teacher probability rows do not match the soft subset")
    student_log_probabilities = F.log_softmax(student_logits, dim=1)
    soft_loss_sum = -(
        teacher_probabilities * student_log_probabilities[soft_mask]
    ).sum()
    hard_mask = ~soft_mask
    hard_loss_sum = (
        F.cross_entropy(
            student_logits[hard_mask],
            true_targets[hard_mask],
            reduction="sum",
        )
        if hard_mask.any()
        else student_logits.sum() * 0.0
    )
    return (soft_loss_sum + hard_loss_sum) / len(true_targets)


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    per_class_correct = torch.zeros(CLASSES, dtype=torch.long)
    per_class_total = torch.zeros(CLASSES, dtype=torch.long)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets_device = targets.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += F.cross_entropy(logits, targets_device, reduction="sum").item()
        prediction = logits.argmax(dim=1).cpu()
        correct_mask = prediction.eq(targets)
        correct += correct_mask.sum().item()
        total += len(targets)
        per_class_total.scatter_add_(0, targets, torch.ones_like(targets))
        per_class_correct.scatter_add_(0, targets, correct_mask.long())
    return {
        "loss": loss_sum / total,
        "top1": 100.0 * correct / total,
        "images": total,
        "per_class": [
            {
                "class_id": class_id,
                "correct": int(per_class_correct[class_id]),
                "total": int(per_class_total[class_id]),
                "top1": 100.0 * int(per_class_correct[class_id]) / int(per_class_total[class_id]),
            }
            for class_id in range(CLASSES)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--test-dir", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--supervision",
        choices=("hard", "soft1", "kd4", "high_entropy_soft1", "low_entropy_soft1"),
        required=True,
    )
    parser.add_argument("--student-seed", choices=STUDENT_SEEDS, required=True, type=int)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int)
    parser.add_argument("--epochs", default=TRAIN_EPOCHS, type=int)
    parser.add_argument("--eval-every-epochs", default=EVAL_EVERY_EPOCHS, type=int)
    parser.add_argument("--eval-epochs", nargs="+", type=int)
    parser.add_argument("--protocol-name", default="imagenette_entropy_selection_v1")
    parser.add_argument(
        "--protocol-spec",
        type=Path,
        default=Path(__file__).resolve().with_name(
            "imagenette_entropy_selection_protocol.json"
        ),
    )
    args = parser.parse_args()
    if args.batch_size != BATCH_SIZE or args.epochs != TRAIN_EPOCHS:
        raise RuntimeError("frozen protocol requires batch64 and 2000 epochs")
    if args.eval_epochs is None:
        if args.eval_every_epochs <= 0 or args.epochs % args.eval_every_epochs:
            raise RuntimeError("evaluation interval must divide 2000 epochs")
        evaluation_epochs = list(
            range(args.eval_every_epochs, args.epochs + 1, args.eval_every_epochs)
        )
    else:
        evaluation_epochs = sorted(set(args.eval_epochs))
        if (
            len(evaluation_epochs) != len(args.eval_epochs)
            or not evaluation_epochs
            or evaluation_epochs[-1] != args.epochs
            or any(epoch <= 0 or epoch > args.epochs for epoch in evaluation_epochs)
        ):
            raise RuntimeError(
                "explicit evaluation epochs must be unique, valid, and include epoch2000"
            )
    evaluation_epoch_set = set(evaluation_epochs)
    protocol_spec = args.protocol_spec.resolve()
    if not protocol_spec.is_file():
        raise RuntimeError(f"missing protocol specification: {protocol_spec}")
    seed_everything(identity_seed("imagenette-training-global-v1", args.student_seed))
    device = torch.device("cuda")

    # Student initialization owns a dedicated RNG namespace and occurs before
    # dataset iteration or Teacher construction. Hard and Soft therefore begin
    # from exactly the same parameters for a given Student seed.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(identity_seed("imagenette-student-init-v1", args.student_seed))
        student = build_student()
    initial_student_sha256 = state_dict_sha256(student.state_dict())

    train_dataset = PairedManifestDataset(args.train_manifest)
    train_dataset.bind_student_seed(args.student_seed)
    test_dataset = datasets.ImageFolder(args.test_dir.resolve(), transform=test_transform)
    if len(test_dataset) != TEST_IMAGES or len(test_dataset.classes) != CLASSES:
        raise RuntimeError("test split is not official 3925-image ImageNette")
    manifest_classes = sorted({row["class_name"] for row in train_dataset.rows})
    if manifest_classes != test_dataset.classes:
        raise RuntimeError("selection/test class order mismatch")
    split_file_list_path = Path(
        train_dataset.manifest["official_split_file_list"]
    )
    split_file_list = json.loads(split_file_list_path.read_text(encoding="utf-8"))
    if file_sha256(split_file_list_path) != train_dataset.manifest[
        "official_split_file_list_sha256"
    ]:
        raise RuntimeError("frozen official split file list changed")
    current_test = [
        {
            "relative_path": Path(path).relative_to(args.test_dir.resolve().parent).as_posix(),
            "class_id": int(target),
            "size_bytes": Path(path).stat().st_size,
        }
        for path, target in test_dataset.samples
    ]
    if current_test != split_file_list["test"]:
        raise RuntimeError("current test split differs from frozen official file list")
    allowed_train = {row["relative_path"] for row in split_file_list["train"]}
    if any(row["relative_path"] not in allowed_train for row in train_dataset.rows):
        raise RuntimeError("selection contains an image outside the frozen full training pool")
    loader_generator = torch.Generator().manual_seed(
        identity_seed("imagenette-shuffle-v1", args.student_seed)
    )
    persistent = args.workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=args.workers,
        persistent_workers=persistent,
        pin_memory=True,
        prefetch_factor=2 if args.workers > 0 else None,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=persistent,
        pin_memory=True,
        prefetch_factor=2 if args.workers > 0 else None,
        drop_last=False,
    )

    student = student.to(device)
    teacher = None
    if args.supervision != "hard":
        teacher = build_teacher(args.teacher_checkpoint.resolve()).to(device).eval()
        if teacher.training or any(parameter.requires_grad for parameter in teacher.parameters()):
            raise RuntimeError("Teacher must remain eval/frozen")
    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    training_history = []
    teacher_sums = {
        "entropy": 0.0,
        "maximum_probability": 0.0,
        "true_class_probability": 0.0,
        "argmax_matches_true_class": 0.0,
    }
    teacher_views = 0
    updates = 0
    best_top1 = float("-inf")
    best_epoch = None
    final_metrics = None
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        lr = lr_for_epoch(epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        student.train()
        epoch_loss_sum = 0.0
        epoch_examples = 0
        epoch_batch_sizes = []
        for raw_views, true_targets, high_entropy_group in train_loader:
            raw_views = raw_views.to(device, non_blocking=True)
            true_targets = true_targets.to(device, non_blocking=True)
            high_entropy_group = high_entropy_group.to(device, non_blocking=True).bool()
            images = normalize(raw_views)
            optimizer.zero_grad(set_to_none=True)
            student_logits = student(images)
            if args.supervision == "hard":
                # F.cross_entropy uses the current batch's actual denominator:
                # 64 for batch one and 36 for batch two.
                loss = F.cross_entropy(student_logits, true_targets)
            else:
                if args.supervision == "high_entropy_soft1":
                    soft_mask = high_entropy_group
                elif args.supervision == "low_entropy_soft1":
                    soft_mask = ~high_entropy_group
                else:
                    soft_mask = torch.ones_like(high_entropy_group)
                if soft_mask.sum().item() == 0:
                    raise RuntimeError("soft-supervised subset is empty in a training batch")
                soft_targets = true_targets[soft_mask]
                with torch.no_grad():
                    teacher_logits = teacher(images[soft_mask])
                    teacher_temperature = 4.0 if args.supervision == "kd4" else 1.0
                    log_q = F.log_softmax(
                        teacher_logits.float() / teacher_temperature, dim=1
                    )
                    teacher_probabilities = log_q.exp()
                    entropy = -(teacher_probabilities * log_q).sum(dim=1)
                    confidence, prediction = teacher_probabilities.max(dim=1)
                    true_probability = teacher_probabilities.gather(
                        1, soft_targets[:, None]
                    ).squeeze(1)
                    teacher_sums["entropy"] += entropy.sum().item()
                    teacher_sums["maximum_probability"] += confidence.sum().item()
                    teacher_sums["true_class_probability"] += true_probability.sum().item()
                    teacher_sums["argmax_matches_true_class"] += prediction.eq(soft_targets).sum().item()
                    teacher_views += len(soft_targets)
                if args.supervision == "soft1":
                    loss = -(
                        teacher_probabilities * F.log_softmax(student_logits, dim=1)
                    ).sum(dim=1).mean()
                elif args.supervision in ("high_entropy_soft1", "low_entropy_soft1"):
                    loss = mixed_hard_soft1_loss(
                        student_logits,
                        true_targets,
                        soft_mask,
                        teacher_probabilities,
                    )
                else:
                    loss = 16.0 * F.kl_div(
                        F.log_softmax(student_logits / 4.0, dim=1),
                        teacher_probabilities,
                        reduction="batchmean",
                    )
            loss.backward()
            optimizer.step()
            updates += 1
            epoch_loss_sum += loss.item() * len(true_targets)
            epoch_examples += len(true_targets)
            epoch_batch_sizes.append(len(true_targets))
        if epoch_batch_sizes != [64, 36]:
            raise RuntimeError(f"expected actual batch sizes [64,36], found {epoch_batch_sizes}")
        if epoch_examples != TRAIN_SIZE or updates != 2 * epoch:
            raise RuntimeError("IPC10 epoch/update cardinality changed")
        if epoch in evaluation_epoch_set:
            metrics = validate(student, test_loader, device)
            if epoch == TRAIN_EPOCHS:
                final_metrics = metrics
            record = {
                "epoch": epoch,
                "update": updates,
                "lr": lr,
                "train_loss": epoch_loss_sum / epoch_examples,
                "test_loss": metrics["loss"],
                "test_top1": metrics["top1"],
            }
            history.append(record)
            training_history.append(
                {
                    "epoch": epoch,
                    "update": updates,
                    "loss": epoch_loss_sum / epoch_examples,
                }
            )
            if metrics["top1"] > best_top1:
                best_top1 = metrics["top1"]
                best_epoch = epoch
            print(json.dumps(record, sort_keys=True), flush=True)

    if (
        updates != TOTAL_UPDATES
        or history[-1]["epoch"] != TRAIN_EPOCHS
        or final_metrics is None
    ):
        raise RuntimeError("training did not reach Final@4000")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint_dir / "final.pth.tar"
    torch.save(
        {
            "protocol": args.protocol_name,
            "supervision": args.supervision,
            "student_seed": args.student_seed,
            "epoch": TRAIN_EPOCHS,
            "update": updates,
            "state_dict": student.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        checkpoint,
    )
    manifest = train_dataset.manifest
    student_source = Path(__file__).resolve().parents[2] / "CoDA" / "test" / "resnet_ap.py"
    payload = {
        "status": "complete",
        "protocol": args.protocol_name,
        "protocol_spec": str(protocol_spec.resolve()),
        "protocol_spec_sha256": file_sha256(protocol_spec),
        "dataset": "imagenet-nette",
        "classes": CLASSES,
        "ipc": IPC,
        "train_images": TRAIN_SIZE,
        "test_images": TEST_IMAGES,
        "lambda": manifest["lambda"],
        "selection_seed": manifest["selection_seed"],
        "student_seed": args.student_seed,
        "supervision": args.supervision,
        "student_model": "CoDA_released_ResNetAP10",
        "student_model_source": str(student_source.resolve()),
        "student_model_source_sha256": file_sha256(student_source),
        "student_initialization": "random",
        "initial_student_state_sha256": initial_student_sha256,
        "normalization_layer": "GroupNorm(C,C)_released_instance_semantics",
        "image_size": IMAGE_SIZE,
        "optimizer": "SGD",
        "initial_lr": LR,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "weight_decay_exclusions": [],
        "batch_size": BATCH_SIZE,
        "actual_batch_sizes_per_epoch": [64, 36],
        "loss_batch_denominator": "current_actual_batch_size",
        "gradient_accumulation_steps": 1,
        "drop_last": False,
        "epochs": TRAIN_EPOCHS,
        "updates_completed": updates,
        "scheduler": "released_multistep_epoch",
        "lr_milestones_after_epochs": [1333, 1666],
        "lr_gamma": 0.2,
        "warmup": False,
        "label_smoothing": 0.0,
        "cutmix": False,
        "mixup": False,
        "train_transform": "deterministic-keyed released Resize256+CenterCrop256+RRC256(0.5,1)+Flip+ColorJitter0.4+Lighting0.1+ImageNetNorm",
        "test_transform": "ResizeShorter256+CenterCrop256+ImageNetNorm",
        "paired_augmentation_key": "student_seed x epoch x image identity",
        "training_sample_weighting": "equal",
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": file_sha256(args.teacher_checkpoint.resolve()),
        "teacher_mode": None if teacher is None else "eval_frozen_no_grad",
        "teacher_and_student_view_pixels_identical": args.supervision != "hard",
        "teacher_temperature": (
            None if args.supervision == "hard" else (4.0 if args.supervision == "kd4" else 1.0)
        ),
        "student_temperature": 4.0 if args.supervision == "kd4" else 1.0,
        "temperature_squared_multiplier": args.supervision == "kd4",
        "loss": {
            "hard": "mean_cross_entropy_true_one_hot",
            "soft1": "mean_soft_target_cross_entropy",
            "kd4": "16_times_KL_teacher_to_student_batchmean",
            "high_entropy_soft1": "batchmean_mixed_soft1_on_high_entropy_half_hard_ce_on_low_entropy_half",
            "low_entropy_soft1": "batchmean_mixed_hard_ce_on_high_entropy_half_soft1_on_low_entropy_half",
        }[args.supervision],
        "soft_label_allocation": {
            "hard": "none",
            "soft1": "all_images",
            "kd4": "all_images",
            "high_entropy_soft1": "top5_calibration_entropy_within_each_class",
            "low_entropy_soft1": "bottom5_calibration_entropy_within_each_class",
        }[args.supervision],
        "soft_supervised_images_per_epoch": {
            "hard": 0,
            "soft1": TRAIN_SIZE,
            "kd4": TRAIN_SIZE,
            "high_entropy_soft1": TRAIN_SIZE // 2,
            "low_entropy_soft1": TRAIN_SIZE // 2,
        }[args.supervision],
        "primary_metric": "final_top1_at_update_4000",
        "evaluation_epochs": evaluation_epochs,
        "evaluation_count": len(evaluation_epochs),
        "final_top1": final_metrics["top1"],
        "final_loss": final_metrics["loss"],
        "best_top1_diagnostic": best_top1,
        "best_epoch_diagnostic": best_epoch,
        "best_update_diagnostic": 2 * best_epoch,
        "test_history": history,
        "per_class_final": final_metrics["per_class"],
        "teacher_online_label_statistics": (
            None
            if teacher_views == 0
            else {
                "views": teacher_views,
                **{f"mean_{key}": value / teacher_views for key, value in teacher_sums.items()},
            }
        ),
        "train_manifest": str(args.train_manifest.resolve()),
        "train_manifest_sha256": file_sha256(args.train_manifest.resolve()),
        "final_checkpoint": str(checkpoint.resolve()),
        "final_checkpoint_sha256": file_sha256(checkpoint),
        "workers": args.workers,
        "persistent_workers": persistent,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(args.result, payload)
    print(json.dumps({"result": str(args.result), "final_top1": payload["final_top1"]}))


if __name__ == "__main__":
    main()
