"""Summarize CUB IPC3 LGM/Random hard-label replay against paired Soft-v2."""

import argparse
import json
import os
import statistics
from pathlib import Path


def stats(values):
    values = list(map(float, values))
    return {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "values": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    rows, errors = [], []
    paths = [("lgm", None, seed, args.root / f"results/lgm/ipc3_sseed{seed}.json") for seed in (42,43,44)]
    paths += [("random_real", selection, seed,
               args.root / f"results/random_real/rseed{selection}/ipc3_sseed{seed}.json")
              for selection in (0,1,2) for seed in (42,43,44)]
    for method, source_seed, seed, path in paths:
        try:
            hard = json.loads(path.read_text(encoding="utf-8")); meta = hard["hard_replay_v2"]
            soft = json.loads(Path(meta["source_soft_result"]).read_text(encoding="utf-8"))
            rows.append({"method": method, "source_seed": source_seed, "student_seed": seed,
                         "hard_best": hard["best_top1"], "hard_final": hard["final_epoch_top1"],
                         "soft_best": soft["best_top1"], "soft_final": soft["final_epoch_top1"],
                         "hard_minus_soft_best": hard["best_top1"]-soft["best_top1"],
                         "hard_minus_soft_final": hard["final_epoch_top1"]-soft["final_epoch_top1"],
                         "best_epoch": hard["best_epoch"], "initial_model_sha256": hard["initial_model_sha256"],
                         "result": str(path.resolve())})
        except Exception as error:
            errors.append({"path": str(path), "error": str(error)})
    groups=[]
    for method in ("lgm","random_real"):
        selected=[row for row in rows if row["method"]==method]
        groups.append({"method":method,"count":len(selected),
                       **{field:stats(row[field] for row in selected) for field in
                          ("hard_best","hard_final","soft_best","soft_final","hard_minus_soft_best","hard_minus_soft_final")},
                       "best_epochs":[row["best_epoch"] for row in selected]})
    hashes={}
    for seed in (42,43,44):
        unique=sorted({row["initial_model_sha256"] for row in rows if row["student_seed"]==seed})
        hashes[str(seed)]={"unique_hashes":len(unique),"hashes":unique}
        if rows and len(unique)!=1: errors.append({"student_seed":seed,"error":"initial hashes differ"})
    payload={"status":"complete" if len(rows)==12 and not errors else "failed",
             "dataset":"CUB_imsize224","ipc":3,"protocol":"standard_v2_hard_fkd_replay",
             "initial_model_hash_audit":hashes,"groups":groups,"rows":rows,"errors":errors}
    output=args.root/"summary/cub_lgm_random_hard_replay_v2.json"; output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(".json.tmp"); temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); os.replace(temporary,output)
    print(json.dumps({"status":payload["status"],"rows":len(rows),"errors":errors},indent=2))
    if payload["status"]!="complete": raise SystemExit(1)


if __name__=="__main__": main()
