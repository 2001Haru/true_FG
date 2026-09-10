"""Summarize the actual T20 entropy of every saved FKD augmented-view logit."""

import argparse
import json
import math
import os
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fkd-dir", required=True, type=Path)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument("--temperature", default=20.0, type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    values = []
    epoch_means = []
    rows = 0
    for epoch in range(args.epochs):
        epoch_values = []
        files = sorted(
            (args.fkd_dir / f"epoch_{epoch}").glob("batch_*.tar"),
            key=lambda path: int(path.stem.split("_")[1]),
        )
        if not files:
            raise RuntimeError(f"missing FKD files for epoch {epoch}")
        for path in files:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            logits = payload[5].float()
            log_probabilities = torch.log_softmax(logits / args.temperature, dim=1)
            entropy = -(log_probabilities.exp() * log_probabilities).sum(dim=1)
            epoch_values.append(entropy)
            rows += logits.shape[0]
        epoch_tensor = torch.cat(epoch_values)
        values.append(epoch_tensor)
        epoch_means.append(float(epoch_tensor.mean()))
    tensor = torch.cat(values)
    quantile_points = torch.tensor([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    quantiles = torch.quantile(tensor, quantile_points)
    result = {
        "status": "complete",
        "fkd_dir": str(args.fkd_dir.resolve()),
        "epochs": args.epochs,
        "views": rows,
        "temperature": args.temperature,
        "log_base": "natural",
        "classes": int(torch.load(
            args.fkd_dir / "epoch_0/batch_0.tar", map_location="cpu", weights_only=False
        )[5].shape[1]),
        "maximum_entropy": math.log(100),
        "mean": float(tensor.mean()),
        "sample_std": float(tensor.std(unbiased=True)),
        "normalized_mean": float(tensor.mean() / math.log(100)),
        "quantiles": {str(float(point)): float(value) for point, value in zip(quantile_points, quantiles)},
        "epoch_mean_min": min(epoch_means),
        "epoch_mean_max": max(epoch_means),
        "epoch_means": epoch_means,
    }
    output = args.output or args.fkd_dir / "t20_entropy_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({key: value for key, value in result.items() if key != "epoch_means"}, sort_keys=True))


if __name__ == "__main__":
    main()
