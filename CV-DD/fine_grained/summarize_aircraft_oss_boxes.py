"""Merge disjoint OSS bbox prediction shards and compute one global summary."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_aircraft_oss_boxes import distribution, summarize


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--shard", action="append", required=True, type=Path)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--output", required=True, type=Path)
    args=parser.parse_args(); rows=[]; configurations=[]
    for root in args.shard:
        configurations.append(json.loads((root/"summary.json").read_text()))
        rows.extend(json.loads(line) for line in (root/"predictions.jsonl").read_text().splitlines())
    identities=[row["identity"] for row in rows]
    if len(rows)!=args.expected_images or len(set(identities))!=len(rows):
        raise RuntimeError((len(rows),len(set(identities)),args.expected_images))
    rows.sort(key=lambda row:row["identity"])
    methods=("grounding_dino","grounding_dino_sam21")
    summary={method:summarize(rows,method) for method in methods}
    delta=[row[methods[1]]["metrics"]["iou"]-row[methods[0]]["metrics"]["iou"] for row in rows]
    summary["sam21_minus_dino"]={"iou_delta":distribution(delta),
      "improved_fraction":float(np.mean(np.asarray(delta)>0)),
      "degraded_fraction":float(np.mean(np.asarray(delta)<0))}
    frozen=("protocol","caption","selection","sam","official_boxes","grounding_revision","sam_revision",
            "grounding_checkpoint_sha256","sam_checkpoint_sha256","text_encoder_root")
    base={key:configurations[0][key] for key in frozen}
    for configuration in configurations[1:]:
        assert all(configuration[key]==base[key] for key in frozen)
    output={"status":"complete",**base,"images":len(rows),"summary":summary,
            "shards":[str(path.resolve()) for path in args.shard]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
