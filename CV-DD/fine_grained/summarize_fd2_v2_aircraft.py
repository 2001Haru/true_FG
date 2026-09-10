"""Summarize and pair the Aircraft FD2 dual-semantics Student-v2 rerun."""

import argparse
import json
import math
import os
import statistics
from pathlib import Path

from scipy.stats import t
from audit_result import audit_payload


SEMANTICS = ("released_semantics", "paper_literal")
IPCS = (1, 3, 5)
SEEDS = (42, 43, 44)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values):
    values = list(map(float, values)); mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    critical = float(t.ppf(.975, len(values)-1)) if len(values) > 1 else 0.0
    half = critical * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "sample_std": sd, "ci95_paired_t": [mean-half, mean+half],
            "positive": sum(x > 0 for x in values), "negative": sum(x < 0 for x in values), "values": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", required=True, type=Path)
    parser.add_argument("--v1-root", required=True, type=Path)
    parser.add_argument("--plain-v2-root", required=True, type=Path)
    args = parser.parse_args(); rows=[]; errors=[]
    for semantics in SEMANTICS:
        for ipc in IPCS:
            for seed in SEEDS:
                path=args.v2_root/semantics/"results/A_imsize224/rseed42"/f"ipc{ipc}_sseed{seed}.json"
                old=args.v1_root/semantics/"results/A_imsize224/rseed42"/f"ipc{ipc}_sseed{seed}.json"
                plain=args.plain_v2_root/"results/tseed42/A_imsize224/rseed42"/f"ipc{ipc}_sseed{seed}.json"
                try:
                    p,o,b=load(path),load(old),load(plain)
                    audit_payload(p,100,3333); audit_payload(o,100,3333); audit_payload(b,100,3333)
                    if p["standard_protocol"]["version"]!="v2" or p["fd2_semantics"]!=semantics: raise RuntimeError("protocol mismatch")
                    rows.append({"semantics":semantics,"ipc":ipc,"student_seed":seed,
                        "best":p["best_top1"],"final":p["final_epoch_top1"],
                        "best_delta_vs_fd2_v1":p["best_top1"]-o["best_top1"],
                        "final_delta_vs_fd2_v1":p["final_epoch_top1"]-o["final_epoch_top1"],
                        "best_delta_vs_plain_v2":p["best_top1"]-b["best_top1"],
                        "final_delta_vs_plain_v2":p["final_epoch_top1"]-b["final_epoch_top1"],"result":str(path.resolve())})
                except Exception as error: errors.append({"result":str(path.resolve()),"error":str(error)})
    groups=[]
    for semantics in SEMANTICS:
        for ipc in IPCS:
            selected=[r for r in rows if r["semantics"]==semantics and r["ipc"]==ipc]
            groups.append({"semantics":semantics,"ipc":ipc,"count":len(selected),
                "best":stats(r["best"] for r in selected),"final":stats(r["final"] for r in selected),
                "best_delta_vs_fd2_v1":stats(r["best_delta_vs_fd2_v1"] for r in selected),
                "final_delta_vs_fd2_v1":stats(r["final_delta_vs_fd2_v1"] for r in selected),
                "best_delta_vs_plain_v2":stats(r["best_delta_vs_plain_v2"] for r in selected),
                "final_delta_vs_plain_v2":stats(r["final_delta_vs_plain_v2"] for r in selected)})
    payload={"status":"complete" if len(rows)==18 and not errors else "failed","expected":18,
             "completed":len(rows),"groups":groups,"rows":rows,"errors":errors}
    output=args.v2_root/"summary/aircraft_dual_semantics_v2.json"; output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(".json.tmp"); temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); os.replace(temporary,output)
    print(json.dumps({"status":payload["status"],"completed":len(rows),"errors":len(errors),"output":str(output.resolve())}))
    if payload["status"]!="complete": raise SystemExit(1)


if __name__=="__main__": main()
