import argparse
import json
import statistics
from pathlib import Path

from audit_result import audit_payload


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser("Summarize CUB ImageNet-student 4k/10k matrix")
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--standard-root", required=True, type=Path)
    parser.add_argument("--four-k-root", required=True, type=Path)
    args = parser.parse_args()
    experiment = args.experiment_root.resolve()
    standard = args.standard_root.resolve()
    four_k = args.four_k_root.resolve()
    rows = []
    missing = []
    for iterations in (4000, 10000):
        for ipc in (1, 3, 5):
            for student_seed in (42, 43, 44):
                result_path = (experiment / "results" / f"iter{iterations}" /
                               "CUB_imsize224/rseed42" /
                               f"ipc{ipc}_sseed{student_seed}.json")
                if not result_path.is_file():
                    missing.append(str(result_path))
                    continue
                payload = load(result_path)
                best = audit_payload(payload, 200, 5794)
                if payload["student_initialization"] != "imagenet-v1":
                    raise RuntimeError(f"wrong Student initialization: {result_path}")
                if payload["student_seed"] != student_seed:
                    raise RuntimeError(f"wrong Student seed: {result_path}")
                expected = {
                    "epochs": 400,
                    "batch_size": 20,
                    "gradient_accumulation_steps": 2,
                    "temperature": 20.0,
                    "optimizer": "adamw",
                    "learning_rate": 1e-3,
                    "weight_decay": 1e-5,
                    "cosine_eta": 2.0,
                    "dataloader_workers": 8,
                    "persistent_workers": True,
                }
                for key, value in expected.items():
                    if payload[key] != value:
                        raise RuntimeError(f"{result_path}: {key}={payload[key]!r} != {value!r}")
                random_path = (
                    four_k / "results/CUB_imsize224/rseed42" /
                    f"ipc{ipc}_sseed{student_seed}.json"
                    if iterations == 4000 else
                    standard / "results/tseed42/CUB_imsize224/rseed42" /
                    f"ipc{ipc}_sseed{student_seed}.json"
                )
                random_payload = load(random_path)
                random_best = audit_payload(random_payload, 200, 5794)
                rows.append({
                    "iterations": iterations,
                    "ipc": ipc,
                    "teacher_seed": 42,
                    "recovery_seed": 42,
                    "student_seed": student_seed,
                    "best_top1": best,
                    "final_epoch_top1": payload["final_epoch_top1"],
                    "random_student_best_top1": random_best,
                    "imagenet_minus_random_student": best - random_best,
                    "result": str(result_path.resolve()),
                    "random_reference": str(random_path.resolve()),
                })
    groups = []
    for iterations in (4000, 10000):
        for ipc in (1, 3, 5):
            group = [r for r in rows if r["iterations"] == iterations and r["ipc"] == ipc]
            best = [r["best_top1"] for r in group]
            final = [r["final_epoch_top1"] for r in group]
            gain = [r["imagenet_minus_random_student"] for r in group]
            groups.append({
                "iterations": iterations,
                "ipc": ipc,
                "completed": len(group),
                "expected": 3,
                "best_mean": statistics.mean(best) if best else None,
                "best_sample_std": statistics.stdev(best) if len(best) > 1 else None,
                "final_mean": statistics.mean(final) if final else None,
                "final_sample_std": statistics.stdev(final) if len(final) > 1 else None,
                "imagenet_minus_random_mean": statistics.mean(gain) if gain else None,
            })
    paired = []
    for ipc in (1, 3, 5):
        deltas = []
        for student_seed in (42, 43, 44):
            four = next((r for r in rows if r["iterations"] == 4000 and
                         r["ipc"] == ipc and r["student_seed"] == student_seed), None)
            ten = next((r for r in rows if r["iterations"] == 10000 and
                        r["ipc"] == ipc and r["student_seed"] == student_seed), None)
            if four and ten:
                deltas.append(four["best_top1"] - ten["best_top1"])
        paired.append({
            "ipc": ipc,
            "completed": len(deltas),
            "expected": 3,
            "mean_delta_4k_minus_10k": statistics.mean(deltas) if deltas else None,
            "sample_std_delta_4k_minus_10k": statistics.stdev(deltas) if len(deltas) > 1 else None,
            "deltas": deltas,
        })
    result = {
        "status": "complete" if len(rows) == 18 else "incomplete",
        "experiment": "cub_imagenet1k_student_iter4000_vs_iter10000",
        "student_initialization": "torchvision IMAGENET1K_V1",
        "teacher_seed": 42,
        "recovery_seed": 42,
        "student_seeds": [42, 43, 44],
        "ipcs": [1, 3, 5],
        "expected_results": 18,
        "completed_results": len(rows),
        "groups": groups,
        "paired_iteration_effect": paired,
        "rows": rows,
        "missing": missing,
    }
    output = experiment / "summary/cub_imagenet_student_matrix.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "completed_results": len(rows),
        "expected_results": 18,
        "output": str(output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
