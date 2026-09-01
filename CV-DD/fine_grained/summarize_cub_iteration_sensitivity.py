import argparse
import hashlib
import json
import math
from pathlib import Path

from audit_result import audit_payload


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expect(actual, expected, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def expect_float(actual, expected: float, label: str) -> None:
    if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def main() -> None:
    parser = argparse.ArgumentParser("Summarize paired CUB 4k-vs-10k recovery sensitivity")
    parser.add_argument("--sensitivity-root", required=True, type=Path)
    parser.add_argument("--standard-root", required=True, type=Path)
    args = parser.parse_args()
    sensitivity = args.sensitivity_root.resolve()
    standard = args.standard_root.resolve()
    teacher = standard / "teachers/CUB_imsize224/tseed42/ResNet18.pth"
    teacher_hash = sha256(teacher)
    recovery_root = sensitivity / "pipeline/recovery/CUB_imsize224/rseed42"
    recovery = load_json(recovery_root / "recovery_manifest.json")
    recovery_audit = load_json(recovery_root / "recovery_output_audit.json")
    expect(recovery["status"], "complete", "Recovery status")
    expect(recovery["recovery_seed"], 42, "Recovery seed")
    expect(recovery["teacher_sha256"], teacher_hash, "Recovery Teacher hash")
    expect(recovery["protocol"]["iterations"], 4000, "Sensitivity iterations")
    expect(recovery_audit["status"], "complete", "Recovery audit")
    expect(recovery_audit["sampling_relation"],
           "IPC1 and IPC3 are byte-identical relative-path subsets of IPC5",
           "Sampling relation")

    rows = []
    missing = []
    for ipc in (1, 3, 5):
        fkd_dir = (sensitivity / "pipeline/fkd/CUB_imsize224/rseed42" /
                   f"ipc{ipc}_bs20_ipc{ipc}")
        result_path = sensitivity / "results/CUB_imsize224/rseed42" / f"ipc{ipc}_sseed42.json"
        baseline_path = (standard / "results/tseed42/CUB_imsize224/rseed42" /
                         f"ipc{ipc}_sseed42.json")
        if not result_path.is_file():
            missing.append(str(result_path))
            continue
        relabel = load_json(fkd_dir / "relabel_manifest.json")
        fkd = load_json(fkd_dir / "fkd_audit.json")
        result = load_json(result_path)
        baseline = load_json(baseline_path)
        expect(relabel["status"], "complete", "Relabel status")
        expect(relabel["ipc"], ipc, "Relabel IPC")
        expect(relabel["teacher_sha256"], teacher_hash, "Relabel Teacher hash")
        expect(relabel["workers"], 8, "Relabel workers")
        expect(relabel["persistent_workers"], False, "Relabel persistent")
        expect(fkd["status"], "complete", "FKD status")
        expect(fkd["images"], 200 * ipc, "FKD images")
        sensitivity_best = audit_payload(result, 200, 5794)
        baseline_best = audit_payload(baseline, 200, 5794)
        for label, payload in (("sensitivity", result), ("baseline", baseline)):
            expect(payload["student_initialization"], "random", f"{label} initialization")
            expect(payload["student_seed"], 42, f"{label} Student seed")
            expect(payload["epochs"], 400, f"{label} epochs")
            expect(payload["batch_size"], 20, f"{label} batch")
            expect(payload["gradient_accumulation_steps"], 2, f"{label} accumulation")
            expect_float(payload["temperature"], 20.0, f"{label} temperature")
            expect(payload["optimizer"], "adamw", f"{label} optimizer")
            expect_float(payload["learning_rate"], 1e-3, f"{label} LR")
            expect_float(payload["weight_decay"], 1e-5, f"{label} weight decay")
            expect_float(payload["cosine_eta"], 2.0, f"{label} eta")
            expect(payload["dataloader_workers"], 8, f"{label} workers")
            expect(payload["persistent_workers"], True, f"{label} persistent")
        rows.append({
            "ipc": ipc,
            "teacher_seed": 42,
            "recovery_seed": 42,
            "student_seed": 42,
            "baseline_iterations": 10000,
            "sensitivity_iterations": 4000,
            "baseline_best_top1": baseline_best,
            "sensitivity_best_top1": sensitivity_best,
            "delta_4k_minus_10k": sensitivity_best - baseline_best,
            "baseline_result": str(baseline_path.resolve()),
            "sensitivity_result": str(result_path.resolve()),
        })
    payload = {
        "status": "complete" if len(rows) == 3 else "incomplete",
        "experiment": "cub_recovery_iterations_4000_vs_10000",
        "comparison": "paired single-variable sensitivity",
        "teacher_seed": 42,
        "recovery_seed": 42,
        "student_seed": 42,
        "teacher_checkpoint": str(teacher.resolve()),
        "teacher_checkpoint_sha256": teacher_hash,
        "standard_root": str(standard),
        "sensitivity_root": str(sensitivity),
        "completed_results": len(rows),
        "expected_results": 3,
        "rows": rows,
        "missing": missing,
    }
    output = sensitivity / "summary/cub_iter4000_vs_10000.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "completed_results": len(rows),
        "output": str(output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
