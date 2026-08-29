import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


def run(deterministic: bool, batch_size: int, classes: int, warmup: int, steps: int) -> dict:
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, classes)
    model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    images = torch.randn(batch_size, 3, 224, 224, device="cuda")
    targets = torch.randint(classes, (batch_size,), device="cuda")
    criterion = nn.CrossEntropyLoss()

    timings = []
    for step in range(warmup + steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if step >= warmup:
            timings.append(elapsed)
    return {
        "deterministic": deterministic,
        "mean_seconds": float(np.mean(timings)),
        "median_seconds": float(np.median(timings)),
        "minimum_seconds": float(np.min(timings)),
        "maximum_seconds": float(np.max(timings)),
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser("Benchmark the exact cuDNN deterministic toggle used by student seeding")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--classes", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    order = (True, False, False, True)
    runs = [
        run(mode, args.batch_size, args.classes, args.warmup, args.steps)
        for mode in order
    ]
    grouped = {
        str(mode).lower(): float(np.mean([
            item["mean_seconds"] for item in runs
            if item["deterministic"] is mode
        ]))
        for mode in (True, False)
    }
    payload = {
        "batch_size": args.batch_size,
        "classes": args.classes,
        "cudnn_benchmark": False,
        "order": list(order),
        "runs": runs,
        "grouped_mean_seconds": grouped,
        "nondeterministic_over_deterministic_speedup": (
            grouped["true"] / grouped["false"]
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
