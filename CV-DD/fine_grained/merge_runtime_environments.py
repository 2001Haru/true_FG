import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser("Merge and audit the two A100 runtime snapshots")
    parser.add_argument("--node-40gb", required=True, type=Path)
    parser.add_argument("--node-80gb", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    nodes = {
        "a100_40gb": read(args.node_40gb),
        "a100_80gb": read(args.node_80gb),
    }
    for label, payload in nodes.items():
        if payload.get("status") != "complete":
            raise RuntimeError(f"incomplete runtime snapshot: {label}")
        if payload.get("node_label") != label:
            raise RuntimeError(f"runtime label mismatch: {label}")

    comparable_fields = (
        ("packages",),
        ("cuda", "torch_cuda"),
        ("cuda", "cudnn"),
        ("required_environment",),
    )
    mismatches = []
    for field_path in comparable_fields:
        values = {}
        for label, payload in nodes.items():
            value = payload
            for part in field_path:
                value = value[part]
            values[label] = value
        if len({json.dumps(value, sort_keys=True) for value in values.values()}) != 1:
            mismatches.append({"field": ".".join(field_path), "values": values})
    if mismatches:
        raise RuntimeError(f"runtime mismatch across nodes: {mismatches}")

    output = {
        "status": "complete",
        "software_stack_identical": True,
        "repository_revisions_recorded_per_node": True,
        "nodes": nodes,
        "audited_equal_fields": [".".join(path) for path in comparable_fields],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "complete",
        "software_stack_identical": True,
        "output": str(args.output.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
