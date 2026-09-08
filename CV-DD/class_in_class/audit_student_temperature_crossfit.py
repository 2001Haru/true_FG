"""Cross-fitted post-hoc temperature scaling for existing final Students."""

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import t
from torch.utils.data import DataLoader
from torchvision import datasets

from imagenette_entropy_protocol import build_student, file_sha256, load_state_dict_payload, test_transform


LAMBDAS = (-4, 0, 32)
SELECTION_SEEDS = (0, 1, 2)
STUDENT_SEEDS = (42, 43, 44)
FOLDS = 5


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stable_digest(*parts):
    return hashlib.sha256("\0".join(map(str, parts)).encode("utf-8")).digest()


def make_folds(dataset, data_root, dataset_name):
    fold = np.empty(len(dataset), dtype=np.int64)
    rows = []
    for class_id in range(len(dataset.classes)):
        indices = [index for index, target in enumerate(dataset.targets) if target == class_id]
        indices.sort(key=lambda index: (
            stable_digest(dataset_name, Path(dataset.samples[index][0]).relative_to(data_root).as_posix()),
            Path(dataset.samples[index][0]).relative_to(data_root).as_posix(),
        ))
        for rank, index in enumerate(indices):
            fold[index] = rank % FOLDS
    digest = hashlib.sha256()
    for index, ((path, target), fold_id) in enumerate(zip(dataset.samples, fold)):
        relative = Path(path).relative_to(data_root).as_posix()
        digest.update(f"{index}\0{relative}\0{target}\0{fold_id}\n".encode("utf-8"))
        rows.append({"index": index, "relative_path": relative, "class_id": int(target), "fold": int(fold_id)})
    counts = [[int(np.sum((np.asarray(dataset.targets) == class_id) & (fold == fold_id))) for fold_id in range(FOLDS)] for class_id in range(len(dataset.classes))]
    return fold, digest.hexdigest(), rows, counts


def tasks(root, task_set="main"):
    output = []
    if task_set == "nontarget_permutation":
        for group in ("high", "low"):
            supervision = f"{group}_entropy_permuted_soft1"
            for selection_seed in SELECTION_SEEDS:
                for student_seed in STUDENT_SEEDS:
                    result = root / "results/lambda0_nontarget_permutation" / f"rseed{selection_seed}" / f"{supervision}_sseed{student_seed}.json"
                    output.append({
                        "condition": supervision, "lambda": 0,
                        "supervision": supervision,
                        "selection_seed": selection_seed,
                        "student_seed": student_seed, "result": result,
                    })
        return output
    for lambda_value in LAMBDAS:
        for supervision in ("hard", "soft1"):
            for selection_seed in SELECTION_SEEDS:
                for student_seed in STUDENT_SEEDS:
                    result = root / f"results/lambda_{lambda_value:+d}/rseed{selection_seed}/{supervision}_sseed{student_seed}.json"
                    output.append({
                        "condition": f"lambda_{lambda_value:+d}_{supervision}", "lambda": lambda_value,
                        "supervision": supervision, "selection_seed": selection_seed,
                        "student_seed": student_seed, "result": result,
                    })
    for supervision in ("high_entropy_soft1", "low_entropy_soft1"):
        for selection_seed in SELECTION_SEEDS:
            for student_seed in STUDENT_SEEDS:
                result = root / f"results/lambda0_entropy_allocation/rseed{selection_seed}/{supervision}_sseed{student_seed}.json"
                output.append({
                    "condition": supervision, "lambda": 0, "supervision": supervision,
                    "selection_seed": selection_seed, "student_seed": student_seed, "result": result,
                })
    return output


def nll(log_temperature, logits, targets):
    temperature = math.exp(float(log_temperature))
    scaled = logits / temperature
    return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(targets)), targets]))


def crossfit(logits, targets, folds):
    scored = np.empty(len(targets), dtype=np.float64)
    temperatures = []
    boundary = []
    for fold_id in range(FOLDS):
        fit = folds != fold_id
        held = folds == fold_id
        result = minimize_scalar(
            nll, args=(logits[fit], targets[fit]), method="bounded",
            bounds=(-6.0, 6.0), options={"xatol": 1e-10, "maxiter": 500},
        )
        if not result.success:
            raise RuntimeError(f"temperature fit failed: {result.message}")
        temperature = math.exp(float(result.x))
        temperatures.append(temperature)
        boundary.append(abs(result.x) > 5.99)
        scaled = logits[held] / temperature
        scored[held] = logsumexp(scaled, axis=1) - scaled[np.arange(int(held.sum())), targets[held]]
    return float(scored.mean()), temperatures, boundary


@torch.inference_mode()
def infer_logits(checkpoint, images, device):
    model = build_student().to(device)
    model.load_state_dict(load_state_dict_payload(checkpoint), strict=True)
    model.eval()
    output = []
    for start in range(0, len(images), 512):
        output.append(model(images[start:start + 512].to(device, non_blocking=True)).float().cpu())
    del model
    return torch.cat(output).numpy().astype(np.float64)


def compute(args):
    dataset = datasets.ImageFolder(args.data_root / "test", transform=test_transform)
    expected = 3925 if args.dataset == "imagenette" else 3929
    if len(dataset) != expected or len(dataset.classes) != 10:
        raise RuntimeError(f"unexpected {args.dataset} test split")
    folds, fold_hash, fold_rows, fold_counts = make_folds(dataset, args.data_root, args.dataset)
    fold_path = args.output_root / "folds.json"
    if not fold_path.exists():
        atomic_json(fold_path, {
            "status": "complete", "dataset": args.dataset, "folds": FOLDS,
            "definition": "within each true class, SHA256(dataset,relative_path) order then round-robin",
            "fold_assignment_sha256": fold_hash, "class_fold_counts": fold_counts,
            "class_to_idx": dataset.class_to_idx, "images": fold_rows,
        })
    elif json.loads(fold_path.read_text(encoding="utf-8"))["fold_assignment_sha256"] != fold_hash:
        raise RuntimeError("fold definition changed")
    print("caching deterministic test tensors", flush=True)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=args.workers, pin_memory=False)
    images = torch.cat([batch for batch, _ in loader]).pin_memory()
    targets = np.asarray(dataset.targets, dtype=np.int64)
    device = torch.device(args.device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    all_tasks = tasks(args.experiment_root, args.task_set)
    for task_index, task in enumerate(all_tasks):
        if task_index % args.shards != args.shard_index:
            continue
        result_payload = json.loads(task["result"].read_text(encoding="utf-8"))
        checkpoint = Path(result_payload["final_checkpoint"])
        output = args.output_root / "models" / f"{task_index:03d}_{task['condition']}_r{task['selection_seed']}_s{task['student_seed']}.json"
        if output.exists() and not args.force:
            print(f"reuse {output.name}", flush=True); continue
        logits = infer_logits(checkpoint, images, device)
        raw_nll = nll(0.0, logits, targets)
        top1 = 100.0 * float(np.mean(logits.argmax(1) == targets))
        if abs(raw_nll - float(result_payload["final_loss"])) > 2e-5:
            raise RuntimeError(f"raw NLL reproduction failed: {task['result']}")
        if abs(top1 - float(result_payload["final_top1"])) > 1e-10:
            raise RuntimeError(f"Top-1 reproduction failed: {task['result']}")
        calibrated_nll, temperatures, boundary = crossfit(logits, targets, folds)
        payload = {
            "status": "complete", "dataset": args.dataset, **{k:v for k,v in task.items() if k != "result"},
            "source_result": str(task["result"]), "source_result_sha256": file_sha256(task["result"]),
            "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
            "fold_assignment_sha256": fold_hash, "raw_final_nll_recomputed": raw_nll,
            "source_final_nll": float(result_payload["final_loss"]),
            "raw_top1_recomputed": top1, "source_top1": float(result_payload["final_top1"]),
            "crossfit_calibrated_nll": calibrated_nll,
            "nll_improvement_raw_minus_calibrated": raw_nll - calibrated_nll,
            "fold_temperatures": temperatures, "temperature_geometric_mean": float(np.exp(np.mean(np.log(temperatures)))),
            "temperature_min": min(temperatures), "temperature_max": max(temperatures),
            "optimizer_log_temperature_bounds": [-6.0, 6.0], "optimizer_hit_boundary": any(boundary),
            "top1_after_temperature_scaling": top1, "top1_changed": False,
            "diagnostic_only": "official test labels used in fixed stratified cross-fitting",
        }
        atomic_json(output, payload)
        print(json.dumps({"model": output.name, "raw": raw_nll, "calibrated": calibrated_nll, "tau": payload["temperature_geometric_mean"]}), flush=True)


def stats(values):
    values = list(map(float, values))
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "values": values}


def crossed_block(rows, key):
    values = [row[key] for row in rows]; mean = statistics.mean(values)
    by_selection = {r: statistics.mean(row[key] for row in rows if row["selection_seed"] == r) for r in SELECTION_SEEDS}
    by_student = {s: statistics.mean(row[key] for row in rows if row["student_seed"] == s) for s in STUDENT_SEEDS}
    residual = [row[key] - by_selection[row["selection_seed"]] - by_student[row["student_seed"]] + mean for row in rows]
    df = 4; mse = sum(value * value for value in residual) / df; se = math.sqrt(mse / 9); critical = float(t.ppf(.975, df))
    return {"mean": mean, "se": se, "df": df, "ci95": [mean-critical*se, mean+critical*se], "two_sided_p": float(2*t.sf(abs(mean/se),df)) if se else 0.0}


def assemble(args):
    all_tasks = tasks(args.experiment_root, args.task_set); models = []
    for index, task in enumerate(all_tasks):
        pattern = args.output_root / "models" / f"{index:03d}_{task['condition']}_r{task['selection_seed']}_s{task['student_seed']}.json"
        if not pattern.exists(): raise RuntimeError(f"missing {pattern}")
        models.append(json.loads(pattern.read_text(encoding="utf-8")))
    fold_hashes = {row["fold_assignment_sha256"] for row in models}
    errors = []
    if len(fold_hashes) != 1: errors.append("fold mismatch")
    if any(row["optimizer_hit_boundary"] or row["top1_changed"] for row in models): errors.append("temperature boundary or Top-1 change")
    conditions = {}
    for condition in sorted({row["condition"] for row in models}):
        group = [row for row in models if row["condition"] == condition]
        conditions[condition] = {
            "models": len(group), "raw_nll": stats(row["raw_final_nll_recomputed"] for row in group),
            "crossfit_calibrated_nll": stats(row["crossfit_calibrated_nll"] for row in group),
            "calibration_gain": stats(row["nll_improvement_raw_minus_calibrated"] for row in group),
            "temperature_geometric_mean": float(np.exp(np.mean([math.log(row["temperature_geometric_mean"]) for row in group]))),
            "temperature_model_range": [min(row["temperature_geometric_mean"] for row in group), max(row["temperature_geometric_mean"] for row in group)],
            "top1": stats(row["source_top1"] for row in group),
        }
    comparisons = {}
    if args.task_set == "nontarget_permutation":
        if args.reference_output_root is None:
            raise RuntimeError("--reference-output-root is required for permutation assembly")
        identity_rows = {}
        for group in ("high", "low"):
            rows = []
            for selection_seed in SELECTION_SEEDS:
                for student_seed in STUDENT_SEEDS:
                    permuted = next(
                        row for row in models
                        if row["condition"] == f"{group}_entropy_permuted_soft1"
                        and row["selection_seed"] == selection_seed
                        and row["student_seed"] == student_seed
                    )
                    matches = list((args.reference_output_root / "models").glob(
                        f"*_{group}_entropy_soft1_r{selection_seed}_s{student_seed}.json"
                    ))
                    if len(matches) != 1:
                        raise RuntimeError(f"expected one original calibration result, found {len(matches)}: {matches}")
                    original = json.loads(matches[0].read_text(encoding="utf-8"))
                    if original["fold_assignment_sha256"] != permuted["fold_assignment_sha256"]:
                        raise RuntimeError("original/permuted fold mismatch")
                    rows.append({
                        "selection_seed": selection_seed,
                        "student_seed": student_seed,
                        "raw": permuted["raw_final_nll_recomputed"] - original["raw_final_nll_recomputed"],
                        "calibrated": permuted["crossfit_calibrated_nll"] - original["crossfit_calibrated_nll"],
                        "permuted_temperature": permuted["temperature_geometric_mean"],
                        "original_temperature": original["temperature_geometric_mean"],
                    })
            identity_rows[group] = rows
            comparisons[f"{group}_permuted_minus_original"] = {
                "sign": "positive NLL means permutation is worse than original identity",
                "raw": stats(row["raw"] for row in rows),
                "calibrated": stats(row["calibrated"] for row in rows),
                "calibrated_crossed_fixed_block": crossed_block(rows, "calibrated"),
                "temperature_ratio_permuted_over_original": stats(
                    row["permuted_temperature"] / row["original_temperature"] for row in rows
                ),
                "tuples": rows,
            }
        contrast = []
        for high, low in zip(identity_rows["high"], identity_rows["low"]):
            if (high["selection_seed"], high["student_seed"]) != (low["selection_seed"], low["student_seed"]):
                raise RuntimeError("high/low tuple order mismatch")
            contrast.append({
                "selection_seed": high["selection_seed"],
                "student_seed": high["student_seed"],
                "raw": high["raw"] - low["raw"],
                "calibrated": high["calibrated"] - low["calibrated"],
            })
        comparisons["high_minus_low_identity_value"] = {
            "definition": "(high permuted - high original) - (low permuted - low original)",
            "raw": stats(row["raw"] for row in contrast),
            "calibrated": stats(row["calibrated"] for row in contrast),
            "calibrated_crossed_fixed_block": crossed_block(contrast, "calibrated"),
            "tuples": contrast,
        }
        payload={
            "status":"complete" if len(models)==18 and not errors else "failed",
            "dataset":args.dataset,"task_set":args.task_set,"models":len(models),"folds":FOLDS,
            "fold_assignment_sha256":next(iter(fold_hashes)),"conditions":conditions,
            "comparisons":comparisons,"errors":errors,
            "reference_output_root":str(args.reference_output_root),
            "diagnostic_only":"official test labels used in fixed 5-fold stratified cross-fitting; original Final metrics remain primary",
        }
        atomic_json(args.output_root / "temperature_crossfit_summary.json",payload)
        print(json.dumps({"status":payload["status"],"models":len(models),"errors":errors},indent=2))
        if payload["status"]!="complete":raise RuntimeError("temperature audit incomplete")
        return
    for lambda_value in LAMBDAS:
        rows = []
        for selection_seed in SELECTION_SEEDS:
            for student_seed in STUDENT_SEEDS:
                hard = next(row for row in models if row["condition"] == f"lambda_{lambda_value:+d}_hard" and row["selection_seed"] == selection_seed and row["student_seed"] == student_seed)
                soft = next(row for row in models if row["condition"] == f"lambda_{lambda_value:+d}_soft1" and row["selection_seed"] == selection_seed and row["student_seed"] == student_seed)
                rows.append({"selection_seed":selection_seed,"student_seed":student_seed,"raw":soft["raw_final_nll_recomputed"]-hard["raw_final_nll_recomputed"],"calibrated":soft["crossfit_calibrated_nll"]-hard["crossfit_calibrated_nll"]})
        comparisons[f"lambda_{lambda_value:+d}_soft_minus_hard"] = {"raw": stats(row["raw"] for row in rows), "calibrated": stats(row["calibrated"] for row in rows), "calibrated_crossed_fixed_block": crossed_block(rows,"calibrated")}
    rows=[]
    for selection_seed in SELECTION_SEEDS:
        for student_seed in STUDENT_SEEDS:
            high=next(row for row in models if row["condition"]=="high_entropy_soft1" and row["selection_seed"]==selection_seed and row["student_seed"]==student_seed)
            low=next(row for row in models if row["condition"]=="low_entropy_soft1" and row["selection_seed"]==selection_seed and row["student_seed"]==student_seed)
            rows.append({"selection_seed":selection_seed,"student_seed":student_seed,"raw":high["raw_final_nll_recomputed"]-low["raw_final_nll_recomputed"],"calibrated":high["crossfit_calibrated_nll"]-low["crossfit_calibrated_nll"]})
    comparisons["high_minus_low"]={"raw":stats(row["raw"] for row in rows),"calibrated":stats(row["calibrated"] for row in rows),"calibrated_crossed_fixed_block":crossed_block(rows,"calibrated")}
    payload={"status":"complete" if len(models)==72 and not errors else "failed","dataset":args.dataset,"models":len(models),"folds":FOLDS,"fold_assignment_sha256":next(iter(fold_hashes)),"conditions":conditions,"comparisons":comparisons,"errors":errors,"diagnostic_only":"official test labels used in fixed 5-fold stratified cross-fitting; original Final metrics remain primary"}
    atomic_json(args.output_root / "temperature_crossfit_summary.json",payload)
    print(json.dumps({"status":payload["status"],"models":len(models),"errors":errors},indent=2))
    if payload["status"]!="complete":raise RuntimeError("temperature audit incomplete")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("compute","assemble"),required=True)
    parser.add_argument("--dataset",choices=("imagenette","imagewoof"),required=True)
    parser.add_argument("--experiment-root",required=True,type=Path)
    parser.add_argument("--data-root",required=True,type=Path)
    parser.add_argument("--output-root",required=True,type=Path)
    parser.add_argument("--task-set", choices=("main", "nontarget_permutation"), default="main")
    parser.add_argument("--reference-output-root", type=Path)
    parser.add_argument("--shard-index",default=0,type=int);parser.add_argument("--shards",default=1,type=int)
    parser.add_argument("--workers",default=8,type=int);parser.add_argument("--device",default="cuda");parser.add_argument("--force",action="store_true")
    args=parser.parse_args();args.experiment_root=args.experiment_root.resolve();args.data_root=args.data_root.resolve();args.output_root=args.output_root.resolve();args.reference_output_root = None if args.reference_output_root is None else args.reference_output_root.resolve();torch.set_num_threads(1)
    if args.phase=="compute":compute(args)
    else:assemble(args)


if __name__=="__main__":main()
