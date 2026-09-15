"""Audit Aircraft RandomReal IPC3/5/10/20 prefix nesting."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    report = {"status": "complete", "dataset": "A_imsize224", "selection_seeds": [0, 1, 2],
              "ipcs": [3, 5, 10, 20], "seeds": {}}
    for seed in (0, 1, 2):
        manifests = {}
        for ipc in (3, 5, 10, 20):
            path = a.selection_root / f"manifests/A_imsize224/rseed{seed}/ipc{ipc}.json"
            payload = json.loads(path.read_text())
            if payload.get("status") != "complete" or payload.get("selection_seed") != seed or payload.get("ipc") != ipc:
                raise RuntimeError(f"invalid manifest: {path}")
            rows = sorted(payload["images"], key=lambda row: (int(row["class_id"]), int(row["rank"])))
            if len(rows) != 100 * ipc:
                raise RuntimeError(f"wrong image count: {path}")
            manifests[ipc] = payload
        checks = []
        for smaller, larger in ((3, 5), (5, 10), (10, 20)):
            small_by_class, large_by_class = {}, {}
            for row in manifests[smaller]["images"]:
                small_by_class.setdefault(int(row["class_id"]), []).append(row)
            for row in manifests[larger]["images"]:
                large_by_class.setdefault(int(row["class_id"]), []).append(row)
            mismatches = 0
            for class_id in range(100):
                small = [row["source_path"] for row in sorted(small_by_class[class_id], key=lambda row: row["rank"])]
                large = [row["source_path"] for row in sorted(large_by_class[class_id], key=lambda row: row["rank"])]
                mismatches += small != large[:smaller]
            if mismatches:
                raise RuntimeError(f"rseed{seed} IPC{smaller} is not an exact prefix of IPC{larger}")
            checks.append({"smaller": smaller, "larger": larger, "class_prefix_mismatches": mismatches})
        report["seeds"][str(seed)] = {"checks": checks, "tree_sha256": {
            str(ipc): manifests[ipc]["selected_tree_sha256"] for ipc in (3, 5, 10, 20)}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
