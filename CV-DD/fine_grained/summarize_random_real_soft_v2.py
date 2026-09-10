import argparse,json,os,statistics
from pathlib import Path
from audit_result import audit_payload


def stats(values):
    values=list(map(float,values)); return {"mean":statistics.mean(values),"sample_std":statistics.stdev(values),"values":values}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",required=True,type=Path); args=p.parse_args()
    rows=[]; errors=[]
    for ipc in (1,3,5):
        for selection in (0,1,2):
            for student in (42,43,44):
                path=args.root/"results/A_imsize224"/f"rseed{selection}"/f"ipc{ipc}_sseed{student}.json"
                try:
                    x=json.loads(path.read_text(encoding="utf-8")); audit_payload(x,100,3333)
                    if x["standard_protocol"]["version"]!="v2" or x["random_real_soft_v2"]["teacher_seed"]!=42: raise RuntimeError("protocol mismatch")
                    rows.append({"ipc":ipc,"selection_seed":selection,"student_seed":student,
                                 "best_top1":x["best_top1"],"final_top1":x["final_epoch_top1"],"result":str(path.resolve())})
                except Exception as e: errors.append({"result":str(path.resolve()),"error":str(e)})
    groups=[]
    for ipc in (1,3,5):
        selected=[r for r in rows if r["ipc"]==ipc]
        groups.append({"ipc":ipc,"count":len(selected),"best_top1":stats(r["best_top1"] for r in selected),
                       "final_top1":stats(r["final_top1"] for r in selected),
                       "selection_means":{str(s):statistics.mean(r["best_top1"] for r in selected if r["selection_seed"]==s) for s in (0,1,2)}})
    payload={"status":"complete" if len(rows)==27 and not errors else "failed","expected":27,"completed":len(rows),
             "dataset":"A_imsize224","teacher_seed":42,"selection_seeds":[0,1,2],"student_seeds":[42,43,44],
             "ipcs":[1,3,5],"supervision":"fixed SRe2L++ Teacher42 FKD soft labels","groups":groups,"rows":rows,"errors":errors}
    out=args.root/"summary/random_real_soft_v2.json"; out.parent.mkdir(parents=True,exist_ok=True); tmp=out.with_suffix(".json.tmp"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); os.replace(tmp,out)
    print(json.dumps({"status":payload["status"],"completed":len(rows),"errors":len(errors),"output":str(out.resolve())}))
    if payload["status"]!="complete": raise SystemExit(1)


if __name__=="__main__": main()
