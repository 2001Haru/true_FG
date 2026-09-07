"""Aggregate exact training-view supervision targets for the lambda=0 controls."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from imagenette_entropy_protocol import (
    CLASSES,
    DeterministicReleasedView,
    TRAIN_EPOCHS,
    atomic_json,
    build_teacher,
    file_sha256,
    identity_seed,
    normalize,
)


CONDITIONS = (
    "all_hard",
    "high_entropy_soft1",
    "high_entropy_permuted_soft1",
    "low_entropy_soft1",
    "low_entropy_permuted_soft1",
    "all_soft1",
)


class TrainingViewDataset(Dataset):
    def __init__(self, rows, student_seed, permutations):
        self.rows = rows
        self.student_seed = student_seed
        self.permutations = permutations
        self.view = DeterministicReleasedView()
        high_paths = set()
        for class_id in range(CLASSES):
            class_rows = [row for row in rows if int(row["class_id"]) == class_id]
            ordered = sorted(
                class_rows,
                key=lambda row: (
                    float(row["calibration_entropy"]),
                    row["relative_path"],
                ),
            )
            high_paths.update(row["relative_path"] for row in ordered[5:])
        self.high_paths = high_paths

    def __len__(self):
        return len(self.rows) * TRAIN_EPOCHS

    def __getitem__(self, task_index):
        epoch_index, image_index = divmod(task_index, len(self.rows))
        epoch = epoch_index + 1
        row = self.rows[image_index]
        seed = identity_seed(
            "imagenette-entropy-student-view-v1",
            self.student_seed,
            epoch,
            row["relative_path"],
        )
        with Image.open(row["source_path"]) as image:
            tensor = self.view(image, seed)
        mapping = self.permutations[row["relative_path"]]
        return (
            tensor,
            int(row["class_id"]),
            row["relative_path"] in self.high_paths,
            torch.tensor(mapping, dtype=torch.long),
            image_index,
        )


def matrix_metrics(original, permuted):
    eye = np.eye(CLASSES, dtype=bool)
    delta = permuted - original
    original_columns = original.mean(axis=0)
    permuted_columns = permuted.mean(axis=0)
    centered_delta = delta - delta.mean(axis=0, keepdims=True)
    return {
        "original_non_diagonal": original[~eye].reshape(CLASSES, CLASSES - 1).tolist(),
        "permuted_non_diagonal": permuted[~eye].reshape(CLASSES, CLASSES - 1).tolist(),
        "non_diagonal_delta": delta[~eye].reshape(CLASSES, CLASSES - 1).tolist(),
        "non_diagonal_l1_change": float(np.abs(delta[~eye]).sum()),
        "non_diagonal_frobenius_change": float(np.sqrt(np.square(delta[~eye]).sum())),
        "original_column_mean": original_columns.tolist(),
        "permuted_column_mean": permuted_columns.tolist(),
        "column_mean_delta": (permuted_columns - original_columns).tolist(),
        "column_mean_l1_change": float(np.abs(permuted_columns - original_columns).sum()),
        "column_mean_max_abs_change": float(np.abs(permuted_columns - original_columns).max()),
        "class_conditional_delta_after_column_mean_removal_frobenius": float(
            np.sqrt(np.square(centered_delta[~eye]).sum())
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--workers", default=16, type=int)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    permutation_path = root / "preflight" / "lambda0_nontarget_permutations.json"
    permutation_payload = json.loads(permutation_path.read_text(encoding="utf-8"))
    permutations = {
        path: row["output_class_takes_input_class"]
        for path, row in permutation_payload["permutations"].items()
    }
    teacher = build_teacher(args.teacher_checkpoint.resolve()).cuda().eval()
    tuple_payloads = []
    aggregate_sums = {
        condition: torch.zeros(CLASSES, CLASSES, dtype=torch.float64)
        for condition in CONDITIONS
    }
    aggregate_counts = torch.zeros(CLASSES, dtype=torch.int64)

    with torch.inference_mode():
        for selection_seed in (0, 1, 2):
            manifest_path = root / "manifests" / f"lambda_{0:+d}_rseed{selection_seed}.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            rows = sorted(
                manifest["images"],
                key=lambda row: (int(row["class_id"]), row["relative_path"]),
            )
            for student_seed in (42, 43, 44):
                dataset = TrainingViewDataset(rows, student_seed, permutations)
                loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.workers,
                    persistent_workers=args.workers > 0,
                    pin_memory=True,
                    prefetch_factor=2 if args.workers > 0 else None,
                )
                sums = {
                    condition: torch.zeros(CLASSES, CLASSES, dtype=torch.float64)
                    for condition in CONDITIONS
                }
                counts = torch.zeros(CLASSES, dtype=torch.int64)
                for batch_index, (images, targets, high, mapping, _) in enumerate(loader):
                    images = normalize(images.cuda(non_blocking=True))
                    targets_gpu = targets.cuda(non_blocking=True)
                    high_gpu = high.cuda(non_blocking=True).bool()
                    mapping_gpu = mapping.cuda(non_blocking=True)
                    probabilities = F.softmax(teacher(images).float(), dim=1)
                    permuted = torch.gather(probabilities, 1, mapping_gpu)
                    hard = F.one_hot(targets_gpu, num_classes=CLASSES).float()
                    target_batches = {
                        "all_hard": hard,
                        "high_entropy_soft1": torch.where(high_gpu[:, None], probabilities, hard),
                        "high_entropy_permuted_soft1": torch.where(high_gpu[:, None], permuted, hard),
                        "low_entropy_soft1": torch.where(high_gpu[:, None], hard, probabilities),
                        "low_entropy_permuted_soft1": torch.where(high_gpu[:, None], hard, permuted),
                        "all_soft1": probabilities,
                    }
                    targets_cpu = targets.long()
                    for condition, target_batch in target_batches.items():
                        sums[condition].index_add_(0, targets_cpu, target_batch.double().cpu())
                    counts.index_add_(0, targets_cpu, torch.ones_like(targets_cpu))
                    if batch_index % 100 == 0:
                        print(
                            f"selection={selection_seed} student={student_seed} "
                            f"views={min((batch_index + 1) * args.batch_size, len(dataset))}/{len(dataset)}",
                            flush=True,
                        )
                if not torch.equal(counts, torch.full_like(counts, 20000)):
                    raise RuntimeError("each true class must contribute 20,000 training views")
                matrices = {
                    condition: (value / counts[:, None]).numpy()
                    for condition, value in sums.items()
                }
                for condition, matrix in matrices.items():
                    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-7):
                        raise RuntimeError(f"row-stochastic target matrix failed: {condition}")
                    aggregate_sums[condition] += sums[condition]
                aggregate_counts += counts
                tuple_payloads.append(
                    {
                        "selection_seed": selection_seed,
                        "student_seed": student_seed,
                        "manifest": str(manifest_path.resolve()),
                        "manifest_sha256": file_sha256(manifest_path),
                        "views_per_true_class": counts.tolist(),
                        "matrices": {key: value.tolist() for key, value in matrices.items()},
                        "comparisons": {
                            "high_original_vs_permuted": matrix_metrics(
                                matrices["high_entropy_soft1"],
                                matrices["high_entropy_permuted_soft1"],
                            ),
                            "low_original_vs_permuted": matrix_metrics(
                                matrices["low_entropy_soft1"],
                                matrices["low_entropy_permuted_soft1"],
                            ),
                        },
                    }
                )
    aggregate = {
        condition: (value / aggregate_counts[:, None]).numpy()
        for condition, value in aggregate_sums.items()
    }
    payload = {
        "status": "complete",
        "experiment": "imagenette_lambda0_supervision_target_matrices",
        "definition": "row=true class, column=class receiving target probability; exact 2000-epoch training augmentation streams",
        "conditions": list(CONDITIONS),
        "tuples": tuple_payloads,
        "aggregate_views_per_true_class": aggregate_counts.tolist(),
        "aggregate_matrices": {key: value.tolist() for key, value in aggregate.items()},
        "aggregate_comparisons": {
            "high_original_vs_permuted": matrix_metrics(
                aggregate["high_entropy_soft1"],
                aggregate["high_entropy_permuted_soft1"],
            ),
            "low_original_vs_permuted": matrix_metrics(
                aggregate["low_entropy_soft1"],
                aggregate["low_entropy_permuted_soft1"],
            ),
        },
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
        "teacher_checkpoint_sha256": file_sha256(args.teacher_checkpoint.resolve()),
        "permutation_file": str(permutation_path.resolve()),
        "permutation_file_sha256": file_sha256(permutation_path),
    }
    atomic_json(args.output, payload)
    csv_dir = args.output.parent / "supervision_target_matrices_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for condition, matrix in aggregate.items():
        np.savetxt(
            csv_dir / f"{condition}.csv",
            matrix,
            delimiter=",",
            header=",".join(f"target_class_{index}" for index in range(CLASSES)),
            comments="",
        )
    print(json.dumps({"status": "complete", "tuples": len(tuple_payloads), "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
