"""Create lambda=0 manifests 3--11, ordering templates, and label-level preflight."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import kendalltau, spearmanr
from torch.utils.data import DataLoader, Dataset

from imagenette_entropy_protocol import (
    CALIBRATION_VIEW_SEED, CLASSES, DeterministicReleasedView, HOLDOUT_VIEW_SEED,
    IPC, SELECTION_VIEWS, atomic_json, build_teacher, file_sha256, gumbel_noise,
    identity_seed, normalize, validate_official_split,
)


MANIFEST_SEEDS = tuple(range(3, 12))
TEMPLATE_VIEWS = 64


class MultiViewDataset(Dataset):
    def __init__(self, rows, namespace, phase_seed, views):
        self.rows = rows
        self.namespace = namespace
        self.phase_seed = phase_seed
        self.views = views
        self.transform = DeterministicReleasedView()

    def __len__(self):
        return len(self.rows) * self.views

    def __getitem__(self, task_index):
        image_index, view_index = divmod(task_index, self.views)
        row = self.rows[image_index]
        seed = identity_seed(
            self.namespace, self.phase_seed, row["relative_path"], view_index
        )
        with Image.open(row["source_path"]) as image:
            tensor = self.transform(image, seed)
        return tensor, int(row["class_id"]), image_index, view_index


@torch.inference_mode()
def extract_probabilities(teacher, rows, namespace, phase_seed, views, batch_size, workers):
    dataset = MultiViewDataset(rows, namespace, phase_seed, views)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        persistent_workers=workers > 0, pin_memory=True,
        prefetch_factor=2 if workers > 0 else None,
    )
    output = torch.empty(len(rows), views, CLASSES, dtype=torch.float32)
    for batch_index, (images, _, image_indices, view_indices) in enumerate(loader):
        probabilities = F.softmax(
            teacher(normalize(images.cuda(non_blocking=True))).float(), dim=1
        ).cpu()
        output[image_indices.long(), view_indices.long()] = probabilities
        if batch_index % 50 == 0:
            print(f"{namespace} views={min((batch_index+1)*batch_size,len(dataset))}/{len(dataset)}", flush=True)
    return output


def conditional_non_target(probabilities, target):
    result = probabilities.clone()
    result[..., target] = 0.0
    return result / result.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(result.dtype).tiny)


def order_from_template(template, target):
    return sorted(
        (index for index in range(CLASSES) if index != target),
        key=lambda index: (-float(template[index]), index),
    )


def mapping_for_order(probability, target, destination_order):
    source_order = sorted(
        (index for index in range(CLASSES) if index != target),
        key=lambda index: (-float(probability[index]), index),
    )
    mapping = list(range(CLASSES))
    for destination, source in zip(destination_order, source_order):
        mapping[destination] = source
    return mapping


def apply_mapping(probability, mapping):
    return probability[torch.tensor(mapping, dtype=torch.long)]


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()), "sample_std": float(values.std(ddof=1)),
        "min": float(values.min()), "p05": float(np.quantile(values, .05)),
        "median": float(np.median(values)), "p95": float(np.quantile(values, .95)),
        "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--teacher-checkpoint", required=True, type=Path)
    parser.add_argument("--parent-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--workers", default=16, type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    preflight_path = output_root / "preflight" / "ordering_preflight.json"
    if preflight_path.exists() and not args.force:
        raise RuntimeError(f"refusing overwrite: {preflight_path}")
    data_root = args.data_root.resolve()
    parent_root = args.parent_root.resolve()
    train, _ = validate_official_split(data_root)
    per_image_path = parent_root / "preflight/per_image_teacher_entropy_and_dino.json"
    base_path = parent_root / "preflight/selection_preflight.json"
    rows = json.loads(per_image_path.read_text(encoding="utf-8"))["images"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    expected = [Path(path).relative_to(data_root).as_posix() for path, _ in train.samples]
    if [row["relative_path"] for row in rows] != expected:
        raise RuntimeError("parent per-image rows differ from official train split")
    targets = np.asarray(train.targets)

    manifests = {}
    high_by_manifest = {}
    for selection_seed in MANIFEST_SEEDS:
        noise = np.asarray([gumbel_noise(selection_seed, row["relative_path"]) for row in rows])
        selected = []
        for class_id in range(CLASSES):
            indices = np.flatnonzero(targets == class_id)
            selected.extend(sorted(indices, key=lambda i: (-float(noise[i]), rows[i]["relative_path"]))[:IPC])
        selected_rows = [{**rows[i], "slot": slot % IPC, "selection_score": float(noise[i]), "gumbel_noise": float(noise[i])} for slot, i in enumerate(selected)]
        high_paths = set()
        for class_id in range(CLASSES):
            class_rows = [row for row in selected_rows if int(row["class_id"]) == class_id]
            ordered = sorted(class_rows, key=lambda row: (float(row["calibration_entropy"]), row["relative_path"]))
            high_paths.update(row["relative_path"] for row in ordered[5:])
        manifest = {
            "status": "complete", "experiment": "imagenette_ordering_hierarchy_v1",
            "dataset": "imagenet-nette", "classes": CLASSES, "ipc": IPC,
            "lambda": 0, "selection_seed": selection_seed,
            "selection_method": "teacher_entropy_percentile_gumbel_topk",
            "selection_images": 100, "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
            "teacher_checkpoint_sha256": file_sha256(args.teacher_checkpoint.resolve()),
            "training_sample_weighting": "equal",
            "candidate_pool": "all images of the true class; no Teacher correctness/confidence filter",
            "official_split_file_list": base["official_split_file_list"],
            "official_split_file_list_sha256": base["official_split_file_list_sha256"],
            "class_to_idx": train.class_to_idx, "images": selected_rows,
        }
        path = output_root / "manifests" / f"lambda_+0_rseed{selection_seed}.json"
        atomic_json(path, manifest)
        manifests[str(selection_seed)] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
        high_by_manifest[selection_seed] = high_paths

    unique_high_paths = sorted(set().union(*high_by_manifest.values()))
    row_by_path = {row["relative_path"]: row for row in rows}
    high_rows = [row_by_path[path] for path in unique_high_paths]
    teacher = build_teacher(args.teacher_checkpoint.resolve()).cuda().eval()
    template_probabilities = extract_probabilities(
        teacher, high_rows, "imagenette-ordering-template-view-v1", 2026090703,
        TEMPLATE_VIEWS, args.batch_size, args.workers,
    )
    holdout_probabilities = extract_probabilities(
        teacher, high_rows, "imagenette-selection-holdout-view-v1", HOLDOUT_VIEW_SEED,
        SELECTION_VIEWS, args.batch_size, args.workers,
    )
    templates = {}
    half_exact = []
    half_top1 = []
    half_spearman = []
    half_kendall = []
    for index, row in enumerate(high_rows):
        target = int(row["class_id"])
        conditional = conditional_non_target(template_probabilities[index], target)
        full = conditional.mean(0)
        first = conditional[:32].mean(0)
        second = conditional[32:].mean(0)
        full_order = order_from_template(full, target)
        first_order = order_from_template(first, target)
        second_order = order_from_template(second, target)
        rank_first = {value: rank for rank, value in enumerate(first_order)}
        rank_second = {value: rank for rank, value in enumerate(second_order)}
        classes = [c for c in range(CLASSES) if c != target]
        spear = float(spearmanr([rank_first[c] for c in classes], [rank_second[c] for c in classes]).statistic)
        kendall = float(kendalltau([rank_first[c] for c in classes], [rank_second[c] for c in classes]).statistic)
        exact = first_order == second_order
        top1 = first_order[0] == second_order[0]
        half_exact.append(exact); half_top1.append(top1); half_spearman.append(spear); half_kendall.append(kendall)
        templates[row["relative_path"]] = {
            "class_id": target, "full_order": full_order,
            "first32_order": first_order, "last32_order": second_order,
            "full_conditional_non_target_template": full.tolist(),
            "first32_vs_last32": {"exact_order": exact, "top1_same": top1, "spearman": spear, "kendall_tau": kendall},
        }

    class_orders = {}
    for selection_seed in MANIFEST_SEEDS:
        class_orders[str(selection_seed)] = {}
        for class_id in range(CLASSES):
            paths = sorted(path for path in high_by_manifest[selection_seed] if int(row_by_path[path]["class_id"]) == class_id)
            if len(paths) != 5:
                raise RuntimeError("class template must use exactly five high-entropy images")
            mean = torch.tensor([templates[path]["full_conditional_non_target_template"] for path in paths]).mean(0)
            class_orders[str(selection_seed)][str(class_id)] = {
                "high_entropy_paths": paths,
                "order": order_from_template(mean, class_id),
                "conditional_non_target_template": mean.tolist(),
            }

    audit_values = defaultdict(list)
    overlap_views = 0
    intervention_square_total = 0.0
    invariant_max = defaultdict(float)
    correctness_mismatches = 0
    audited_views = 0
    for selection_seed in MANIFEST_SEEDS:
        for path in sorted(high_by_manifest[selection_seed]):
            index = unique_high_paths.index(path)
            target = int(row_by_path[path]["class_id"])
            image_order = templates[path]["full_order"]
            class_order = class_orders[str(selection_seed)][str(target)]["order"]
            for original in holdout_probabilities[index]:
                bmap = mapping_for_order(original, target, image_order)
                amap = mapping_for_order(original, target, class_order)
                invmap = np.argsort(amap).tolist()
                B = apply_mapping(original, bmap)
                A = apply_mapping(original, amap)
                Aprime = apply_mapping(original, invmap)
                dB = float(torch.linalg.vector_norm(B - original))
                dA = float(torch.linalg.vector_norm(A - original))
                dAp = float(torch.linalg.vector_norm(Aprime - original))
                dAAp = float(torch.linalg.vector_norm(A - Aprime))
                audit_values["O_to_B_l2"].append(dB)
                audit_values["O_to_A_l2"].append(dA)
                audit_values["O_to_Aprime_l2"].append(dAp)
                audit_values["A_to_Aprime_l2"].append(dAAp)
                invariant_max["A_Aprime_distance_difference"] = max(invariant_max["A_Aprime_distance_difference"], abs(dA - dAp))
                same = torch.equal(A, Aprime)
                overlap_views += int(same)
                intervention_square_total += dA * dA
                if same:
                    audit_values["overlap_A_intervention_square"].append(dA * dA)
                else:
                    audit_values["overlap_A_intervention_square"].append(0.0)
                onehot = F.one_hot(torch.tensor(target), num_classes=CLASSES).float()
                for name, changed in (("B", B), ("A", A), ("Aprime", Aprime)):
                    invariant_max[f"{name}_qy"] = max(invariant_max[f"{name}_qy"], abs(float(changed[target] - original[target])))
                    invariant_max[f"{name}_entropy"] = max(invariant_max[f"{name}_entropy"], abs(float(-(changed*changed.clamp_min(1e-30).log()).sum() + (original*original.clamp_min(1e-30).log()).sum())))
                    invariant_max[f"{name}_k2"] = max(invariant_max[f"{name}_k2"], abs(float(1/changed.square().sum() - 1/original.square().sum())))
                    invariant_max[f"{name}_el2n"] = max(invariant_max[f"{name}_el2n"], abs(float(torch.linalg.vector_norm(changed-onehot)-torch.linalg.vector_norm(original-onehot))))
                    invariant_max[f"{name}_max"] = max(invariant_max[f"{name}_max"], abs(float(changed.max()-original.max())))
                    top2_changed = float(changed[target] + torch.cat((changed[:target],changed[target+1:])).max())
                    top2_original = float(original[target] + torch.cat((original[:target],original[target+1:])).max())
                    invariant_max[f"{name}_top2"] = max(invariant_max[f"{name}_top2"], abs(top2_changed-top2_original))
                    correctness_mismatches += int((changed.argmax().item()==target)!=(original.argmax().item()==target))
                audited_views += 1

    overlap_square = sum(audit_values["overlap_A_intervention_square"])
    overlap_rate = overlap_views / audited_views
    overlap_square_fraction = overlap_square / intervention_square_total if intervention_square_total else 1.0
    gate_pass = overlap_square_fraction < 0.5 and distribution(audit_values["A_to_Aprime_l2"])["median"] > 0
    template_payload = {
        "status": "complete", "template_views": TEMPLATE_VIEWS,
        "template_namespace": "imagenette-ordering-template-view-v1",
        "image_templates": templates, "class_templates": class_orders,
        "manifests": manifests,
    }
    template_path = output_root / "preflight" / "ordering_templates.json"
    atomic_json(template_path, template_payload)
    preflight = {
        "status": "complete_passed_training_gate" if gate_pass else "complete_failed_control_gate",
        "experiment": "imagenette_ordering_hierarchy_v1", "manifests": manifests,
        "ordering_templates": str(template_path.resolve()),
        "ordering_templates_sha256": file_sha256(template_path),
        "unique_high_entropy_images": len(high_rows),
        "image_template_half_stability": {
            "images": len(high_rows), "exact_order_rate": statistics.mean(half_exact),
            "top1_same_rate": statistics.mean(half_top1),
            "spearman": distribution(half_spearman), "kendall_tau": distribution(half_kendall),
        },
        "holdout_label_audit": {
            "views": audited_views,
            "O_to_B_l2": distribution(audit_values["O_to_B_l2"]),
            "O_to_A_l2": distribution(audit_values["O_to_A_l2"]),
            "O_to_Aprime_l2": distribution(audit_values["O_to_Aprime_l2"]),
            "A_to_Aprime_l2": distribution(audit_values["A_to_Aprime_l2"]),
            "A_Aprime_exact_label_overlap_rate": overlap_rate,
            "overlap_views_fraction_of_A_total_squared_intervention": overlap_square_fraction,
            "maximum_invariant_differences": dict(invariant_max),
            "teacher_correctness_mismatches_across_B_A_Aprime": correctness_mismatches,
        },
        "training_gate": {
            "passed": gate_pass,
            "rule": "A/Aprime overlap views must contribute <50% of A total squared intervention and median A-to-Aprime L2 must be positive",
        },
        "teacher_checkpoint_sha256": file_sha256(args.teacher_checkpoint.resolve()),
        "parent_per_image_audit_sha256": file_sha256(per_image_path),
        "calibration_view_seed": CALIBRATION_VIEW_SEED,
        "holdout_view_seed": HOLDOUT_VIEW_SEED,
    }
    atomic_json(preflight_path, preflight)
    print(json.dumps({"status": preflight["status"], "gate_passed": gate_pass, "unique_high_images": len(high_rows), "preflight": str(preflight_path)}, indent=2))


if __name__ == "__main__":
    main()
