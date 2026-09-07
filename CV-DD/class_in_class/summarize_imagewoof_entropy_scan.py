"""Summarize the complete ImageWoof lambda and deterministic Top10 scan."""

import argparse
import json
import os
import statistics
from pathlib import Path


LAMBDAS = (-4, -2, 0, 2, 4, 8, 16, 32)
SELECTION_SEEDS = (0, 1, 2)
STUDENT_SEEDS = (42, 43, 44)


def stats(values):
    values = list(map(float, values))
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "values": values}


def atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); root = args.experiment_root.resolve()
    errors = []; summary = {}; found = 0
    for value in LAMBDAS:
        hard = []; soft = []; gains = []
        for selection_seed in SELECTION_SEEDS:
            for student_seed in STUDENT_SEEDS:
                payloads = {}
                for supervision in ("hard", "soft1"):
                    path = root / f"results/lambda_{value:+d}/rseed{selection_seed}/{supervision}_sseed{student_seed}.json"
                    if not path.exists(): errors.append(f"missing {path}"); continue
                    payloads[supervision] = json.loads(path.read_text(encoding="utf-8")); found += 1
                if len(payloads) != 2: continue
                if payloads["hard"]["initial_student_state_sha256"] != payloads["soft1"]["initial_student_state_sha256"]:
                    errors.append(f"initialization mismatch lambda{value} r{selection_seed}s{student_seed}")
                if payloads["hard"]["train_manifest_sha256"] != payloads["soft1"]["train_manifest_sha256"]:
                    errors.append(f"manifest mismatch lambda{value} r{selection_seed}s{student_seed}")
                h=float(payloads["hard"]["final_top1"]);s=float(payloads["soft1"]["final_top1"])
                hard.append(h);soft.append(s);gains.append(s-h)
        if len(hard)==9:
            summary[str(value)]={"hard_final_top1":stats(hard),"soft1_final_top1":stats(soft),"paired_soft1_minus_hard":stats(gains)}
    top = {}
    for supervision in ("hard", "soft1"):
        values=[]
        for student_seed in STUDENT_SEEDS:
            path=root/f"results/highest_entropy_top10/{supervision}_sseed{student_seed}.json"
            if not path.exists():errors.append(f"missing {path}");continue
            values.append(json.loads(path.read_text(encoding="utf-8"))["final_top1"]);found+=1
        if len(values)==3:top[supervision]=stats(values)
    if len(top)==2:
        top["paired_soft1_minus_hard"]=stats(s-h for s,h in zip(top["soft1"]["values"],top["hard"]["values"]))
    expected=8*2*3*3+2*3
    payload={
        "status":"complete" if found==expected and not errors else "incomplete",
        "experiment":"imagewoof_entropy_selection_v1","expected_results":expected,"found_results":found,
        "lambdas":list(LAMBDAS),"summary":summary,"highest_entropy_top10":top,"errors":errors,
        "primary":"Final Top-1 at update4000",
    }
    atomic(args.output.resolve(),payload);print(json.dumps({k:payload[k] for k in ("status","expected_results","found_results","errors")},indent=2))
    if payload["status"]!="complete":raise RuntimeError("ImageWoof entropy scan incomplete")


if __name__ == "__main__":
    main()
