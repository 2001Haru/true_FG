import argparse
import functools
import hashlib
import json
import math
import os
from pathlib import Path

from audit_result import audit_payload


DATASETS = {
    "CUB_imsize224": {"classes": 200, "validation": 5794, "iterations": 10000, "batch": 20},
    "A_imsize224": {"classes": 100, "validation": 3333, "iterations": 4000, "batch": 20},
    "SC_imsize224": {"classes": 196, "validation": 8041, "iterations": 4000, "batch": 14},
}
SEEDS = (42, 43, 44)
IPCS = (1, 3, 5)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=None)
def sha256(path_text: str) -> str:
    path = Path(path_text)
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
    parser = argparse.ArgumentParser("Audit the completed standard SRe2L++ matrix")
    parser.add_argument("--standard-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.standard_root.resolve()

    definition = load_json(root / "matrix_definition.json")
    expect(definition["protocol_name"], "standard_protocol", "definition protocol")
    expect(definition["protocol_version"], "v1", "definition version")
    expect(definition["expected_results"], 243, "definition result count")
    expect(definition["legacy_results_status"], "exploratory_only", "legacy status")

    status_root = root / "status"
    task_complete = sorted(status_root.glob("*_t*_r*.complete"))
    expect(len(task_complete), 27, "complete arm markers")
    expect(len(list(status_root.glob("*.running"))), 0, "running markers")
    expect(len(list(status_root.glob("*.failed"))), 0, "failed markers")
    if not (status_root / "launcher.complete").is_file():
        raise RuntimeError("launcher.complete is missing")

    teacher_count = recovery_count = relabel_count = result_count = 0
    teacher_gate_passed = 0
    teacher_hashes = {}
    recovery_hashes = {}
    relabel_hashes = {}
    fkd_audit_hashes = {}

    for dataset, cfg in DATASETS.items():
        data_audit = load_json(root / "audits" / f"{dataset}.json")
        expect(data_audit["status"], "valid", f"{dataset} data audit")
        expect(data_audit["dataset"]["classes"], cfg["classes"], f"{dataset} classes")
        for teacher_seed in SEEDS:
            teacher_dir = root / "teachers" / dataset / f"tseed{teacher_seed}"
            complete_path = teacher_dir / "complete.json"
            checkpoint_path = teacher_dir / "ResNet18.pth"
            gate_path = teacher_dir / "teacher_gate.json"
            teacher = load_json(complete_path)
            gate = load_json(gate_path)
            expect(teacher["status"], "complete", "Teacher status")
            expect(teacher["dataset"], dataset, "Teacher dataset")
            expect(teacher["seed"], teacher_seed, "Teacher seed")
            expect(teacher["initialization"], "imagenet-v1", "Teacher initialization")
            expect(teacher["epochs"], 100, "Teacher epochs")
            expect(teacher["batch_size"], 32, "Teacher batch")
            expect(teacher["optimizer"], "SGD", "Teacher optimizer")
            expect_float(teacher["lr"], 1e-2, "Teacher LR")
            expect_float(teacher["momentum"], 0.9, "Teacher momentum")
            expect_float(teacher["weight_decay"], 1e-4, "Teacher weight decay")
            expect_float(teacher["eta_min"], 1e-5, "Teacher eta_min")
            expect(teacher["dataloader_workers"], 8, "Teacher workers")
            expect(teacher["persistent_workers"], False, "Teacher persistent")
            checkpoint_hash = sha256(str(checkpoint_path.resolve()))
            expect(gate["teacher_sha256"], checkpoint_hash, "Teacher gate hash")
            teacher_gate_passed += int(bool(gate["passed"]))
            teacher_hashes[(dataset, teacher_seed)] = checkpoint_hash
            teacher_count += 1

            arm_root = root / "arms" / f"tseed{teacher_seed}"
            for recovery_seed in SEEDS:
                recovery_root = arm_root / "recovery" / dataset / f"rseed{recovery_seed}"
                manifest_path = recovery_root / "recovery_manifest.json"
                output_audit_path = recovery_root / "recovery_output_audit.json"
                recovery = load_json(manifest_path)
                recovery_audit = load_json(output_audit_path)
                expect(recovery["status"], "complete", "Recovery status")
                expect(recovery["recovery_seed"], recovery_seed, "Recovery seed")
                expect(recovery["teacher_sha256"], checkpoint_hash, "Recovery Teacher hash")
                protocol = recovery["protocol"]
                expect(protocol["ipc_recovered"], 5, "Recovery IPC")
                expect(protocol["class_batch_size"], 100, "Recovery class batch")
                expect(protocol["optimizer"], "Adam", "Recovery optimizer")
                expect(protocol["betas"], [0.5, 0.9], "Recovery betas")
                expect_float(protocol["image_lr"], 1e-3, "Recovery image LR")
                expect(protocol["iterations"], cfg["iterations"], "Recovery iterations")
                expect_float(protocol["r_bn"], 1e-3, "Recovery R_BN")
                expect_float(protocol["first_bn_multiplier"], 10.0, "Recovery first-BN")
                expect(protocol["jitter"], 32, "Recovery jitter")
                expect(recovery_audit["status"], "complete", "Recovery output audit")
                expect(recovery_audit["sampling_relation"],
                       "IPC1 and IPC3 are byte-identical relative-path subsets of IPC5",
                       "Recovery sampling relation")
                for ipc in IPCS:
                    tree = recovery_audit["trees"][str(ipc)]
                    expect(tree["ipc"], ipc, "Recovery tree IPC")
                    expect(tree["classes"], cfg["classes"], "Recovery tree classes")
                    expect(tree["files"], cfg["classes"] * ipc, "Recovery tree files")
                recovery_hashes[(dataset, teacher_seed, recovery_seed)] = sha256(
                    str(manifest_path.resolve())
                )
                recovery_count += 1

                for ipc in IPCS:
                    fkd_dir = (arm_root / "fkd" / dataset / f"rseed{recovery_seed}" /
                               f"ipc{ipc}_bs{cfg['batch']}_ipc{ipc}")
                    relabel_path = fkd_dir / "relabel_manifest.json"
                    fkd_audit_path = fkd_dir / "fkd_audit.json"
                    relabel = load_json(relabel_path)
                    fkd = load_json(fkd_audit_path)
                    expect(relabel["status"], "complete", "Relabel status")
                    expect(relabel["dataset_name"], dataset, "Relabel dataset")
                    expect(relabel["ipc"], ipc, "Relabel IPC")
                    expect(relabel["teacher_sha256"], checkpoint_hash, "Relabel Teacher hash")
                    expect(relabel["teacher_mode"], "train", "Relabel Teacher mode")
                    expect(relabel["epochs"], 400, "Relabel epochs")
                    expect(relabel["batch_size"], cfg["batch"], "Relabel batch")
                    expect(relabel["workers"], 8, "Relabel workers")
                    expect(relabel["persistent_workers"], False, "Relabel persistent")
                    expect(relabel["prefetch_factor"], 2, "Relabel prefetch")
                    expect(relabel["seed"], 42, "Relabel seed")
                    expect(relabel["fkd_seed"], 42, "Relabel FKD seed")
                    expect(relabel["mix_type"], "cutmix", "Relabel mix")
                    expect_float(relabel["cutmix_alpha"], 1.0, "Relabel CutMix alpha")
                    expect(fkd["status"], "complete", "FKD audit status")
                    expect(fkd["images"], cfg["classes"] * ipc, "FKD images")
                    expect(fkd["classes"], cfg["classes"], "FKD classes")
                    expect(fkd["batch_size"], cfg["batch"], "FKD batch")
                    expect(fkd["epochs"], 400, "FKD epochs")
                    relabel_hashes[(dataset, teacher_seed, recovery_seed, ipc)] = sha256(
                        str(relabel_path.resolve())
                    )
                    fkd_audit_hashes[(dataset, teacher_seed, recovery_seed, ipc)] = sha256(
                        str(fkd_audit_path.resolve())
                    )
                    relabel_count += 1

                    for student_seed in SEEDS:
                        result_path = (root / "results" / f"tseed{teacher_seed}" / dataset /
                                       f"rseed{recovery_seed}" /
                                       f"ipc{ipc}_sseed{student_seed}.json")
                        result = load_json(result_path)
                        audit_payload(result, cfg["classes"], cfg["validation"])
                        expect(result["student_initialization"], "random", "Student initialization")
                        expect(result["student_seed"], student_seed, "Student seed")
                        expect(result["epochs"], 400, "Student epochs")
                        expect(result["batch_size"], cfg["batch"], "Student batch")
                        expect(result["gradient_accumulation_steps"], 2, "Student accumulation")
                        expect_float(result["temperature"], 20.0, "Student temperature")
                        expect(result["optimizer"], "adamw", "Student optimizer")
                        expect_float(result["learning_rate"], 1e-3, "Student LR")
                        expect_float(result["weight_decay"], 1e-5, "Student weight decay")
                        expect_float(result["cosine_eta"], 2.0, "Student eta")
                        expect(result["dataloader_workers"], 8, "Student workers")
                        expect(result["persistent_workers"], True, "Student persistent")
                        standard = result["standard_protocol"]
                        expect(standard["name"], "standard_protocol", "Result protocol")
                        expect(standard["version"], "v1", "Result protocol version")
                        expect(standard["dataset"], dataset, "Result dataset")
                        expect(standard["teacher_seed"], teacher_seed, "Result Teacher seed")
                        expect(standard["recovery_seed"], recovery_seed, "Result Recovery seed")
                        expect(standard["ipc"], ipc, "Result IPC")
                        expect(standard["student_seed"], student_seed, "Result Student seed")
                        expect(standard["teacher_checkpoint_sha256"], checkpoint_hash,
                               "Result Teacher checkpoint hash")
                        expect(standard["recovery_manifest_sha256"],
                               recovery_hashes[(dataset, teacher_seed, recovery_seed)],
                               "Result Recovery manifest hash")
                        expect(standard["relabel_manifest_sha256"],
                               relabel_hashes[(dataset, teacher_seed, recovery_seed, ipc)],
                               "Result Relabel manifest hash")
                        expect(standard["fkd_audit_sha256"],
                               fkd_audit_hashes[(dataset, teacher_seed, recovery_seed, ipc)],
                               "Result FKD audit hash")
                        result_count += 1

    expect(teacher_count, 9, "Teacher count")
    expect(teacher_gate_passed, 9, "Teacher gates passed")
    expect(recovery_count, 27, "Recovery count")
    expect(relabel_count, 81, "Relabel count")
    expect(result_count, 243, "Result count")
    summary_path = root / "summary" / "standard_matrix.json"
    summary = load_json(summary_path)
    expect(summary["status"], "complete", "Summary status")
    expect(summary["completed_results"], 243, "Summary result count")
    expect(summary["invalid_results"], 0, "Summary invalid count")
    payload = {
        "status": "complete",
        "protocol_name": "standard_protocol",
        "protocol_version": "v1",
        "standard_root": str(root),
        "git_revision": definition["git_revision"],
        "data_audits": len(DATASETS),
        "teachers": teacher_count,
        "teacher_gates_passed": teacher_gate_passed,
        "recovery_arms": recovery_count,
        "relabel_fkd_trees": relabel_count,
        "student_results": result_count,
        "task_complete_markers": len(task_complete),
        "running_markers": 0,
        "failed_markers": 0,
        "summary": str(summary_path.resolve()),
        "summary_sha256": sha256(str(summary_path.resolve())),
    }
    output = root / "summary" / "completion_audit.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
