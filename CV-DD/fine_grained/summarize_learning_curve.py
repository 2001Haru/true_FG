import argparse
import json
import re
import statistics
from pathlib import Path


LINE = re.compile(
    r"^(?P<split>TRAIN|TEST) Iter (?P<epoch>\d+): "
    r"(?:lr = [-+0-9.eE]+,\s*)?"
    r"loss = (?P<loss>[-+0-9.eE]+),\s*"
    r"Top-1 err = (?P<top1_error>[-+0-9.eE]+),\s*"
    r"Top-5 err = (?P<top5_error>[-+0-9.eE]+),\s*"
    r"(?P<time_name>train_time|val_time) = (?P<seconds>[-+0-9.eE]+)"
)


def main() -> None:
    parser = argparse.ArgumentParser("Parse a train_fkd log into a learning-curve JSON")
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = {"train": [], "validation": []}
    for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LINE.match(line)
        if not match:
            continue
        row = {
            "epoch": int(match.group("epoch")),
            "loss": float(match.group("loss")),
            "top1": 100.0 - float(match.group("top1_error")),
            "top5": 100.0 - float(match.group("top5_error")),
            "seconds": float(match.group("seconds")),
        }
        key = "train" if match.group("split") == "TRAIN" else "validation"
        records[key].append(row)
    if not records["train"] or not records["validation"]:
        raise RuntimeError(f"missing train/validation records in {args.log}")
    best = max(records["validation"], key=lambda row: row["top1"])
    late = [row for row in records["validation"] if row["epoch"] >= 300]
    late_slope = None
    if len(late) >= 2:
        late_slope = (
            late[-1]["top1"] - late[0]["top1"]
        ) / (late[-1]["epoch"] - late[0]["epoch"])
    payload = {
        "status": "complete",
        "log": str(args.log.resolve()),
        "train_records": len(records["train"]),
        "validation_records": len(records["validation"]),
        "best_validation": best,
        "last_validation": records["validation"][-1],
        "mean_train_epoch_seconds": statistics.mean(
            row["seconds"] for row in records["train"]
        ),
        "total_reported_train_seconds": sum(
            row["seconds"] for row in records["train"]
        ),
        "total_reported_validation_seconds": sum(
            row["seconds"] for row in records["validation"]
        ),
        "validation_top1_slope_per_epoch_from_300": late_slope,
        "curve": records,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
