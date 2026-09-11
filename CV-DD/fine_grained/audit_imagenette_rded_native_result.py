"""Fail closed on a native-RDED ImageNette result JSON."""

import argparse
import json
import math
from pathlib import Path


def expect(actual, expected, label):
    if actual != expected:
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def close(actual, expected, label, tol=1e-10):
    if not math.isclose(float(actual), expected, rel_tol=0, abs_tol=tol):
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--ipc", required=True, choices=(10, 50), type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--arm", required=True, choices=("a_original", "d_rded"))
    args = parser.parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    expect(payload["status"], "complete", "status")
    expect(payload["ipc"], args.ipc, "IPC")
    expect(payload["student_seed"], args.seed, "seed")
    expect(payload["image_arm"], args.arm, "arm")
    expect(payload["validation_images"], 3925, "full validation")
    expect(payload["teacher_mode"], "eval", "Teacher mode")
    expect(payload["student_bn_extra_mixed_no_grad_forward"], True, "native BN forward")
    expect(payload["shuffle_patches"], args.arm == "d_rded", "ShufflePatches")
    expect(payload["rrc_scale"], [0.5, 1.0], "RRC")
    expect(payload["batch_size"], 50 if args.ipc == 10 else 100, "batch")
    expect(payload["gradient_accumulation_steps"], 1, "accumulation")
    expect(payload["optimizer_updates"], 600 if args.ipc == 10 else 1500, "updates")
    close(payload["learning_rate"], 1e-3, "LR")
    close(payload["weight_decay_matrix_weights"], 0.01, "WD")
    close(payload["temperature_teacher"], 20, "Teacher T")
    close(payload["temperature_student"], 20, "Student T")
    expect(payload["temperature_squared_compensation"], False, "T squared")
    expect(payload["scheduler_eta"], 2.0, "eta")
    close(payload["learning_rate_after_final_step"], 5e-4, "final LR", tol=1e-8)
    if len(payload.get("initial_model_sha256", "")) != 64:
        raise RuntimeError("missing initial model hash")
    print(json.dumps({"status": "complete", "result": str(args.result.resolve())}))


if __name__ == "__main__":
    main()
