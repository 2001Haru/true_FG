"""Summarize Aircraft IPC3 RDED/Random × Soft/Hard full-frame no-CutMix matrix."""

import argparse,json,os,statistics
from pathlib import Path

def stats(v):
    v=list(map(float,v));return {"mean":statistics.mean(v),"sample_std":statistics.stdev(v),"values":v}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",required=True,type=Path);a=p.parse_args()
    rows=[];errors=[]
    for method,seeds in (("random_real",(0,1,2)),("original_rded",(42,43,44))):
      for source_seed in seeds:
       for student in (42,43,44):
        paths={label:a.root/f"results/{label}/{method}/source_seed{source_seed}/ipc3_sseed{student}.json" for label in ("soft","hard")}
        try:
         s=json.loads(paths["soft"].read_text());h=json.loads(paths["hard"].read_text())
         if s["initial_model_sha256"]!=h["initial_model_sha256"]:raise RuntimeError("SL/HL initialization mismatch")
         rows.append({"method":method,"source_seed":source_seed,"student_seed":student,
          "soft_best":s["best_top1"],"soft_final":s["final_epoch_top1"],"hard_best":h["best_top1"],"hard_final":h["final_epoch_top1"],
          "hard_minus_soft_best":h["best_top1"]-s["best_top1"],"hard_minus_soft_final":h["final_epoch_top1"]-s["final_epoch_top1"],
          "soft_best_epoch":s["best_epoch"],"hard_best_epoch":h["best_epoch"],"initial_model_sha256":h["initial_model_sha256"]})
        except Exception as e:errors.append({"paths":{k:str(v) for k,v in paths.items()},"error":str(e)})
    groups=[]
    for method in ("random_real","original_rded"):
     z=[r for r in rows if r["method"]==method]
     groups.append({"method":method,"count":len(z),**{k:stats(r[k] for r in z) for k in
      ("soft_best","soft_final","hard_best","hard_final","hard_minus_soft_best","hard_minus_soft_final")},
      "soft_best_epochs":[r["soft_best_epoch"] for r in z],"hard_best_epochs":[r["hard_best_epoch"] for r in z]})
    payload={"status":"complete" if len(rows)==18 and not errors else "failed","dataset":"A_imsize224","ipc":3,
     "view":"full-frame Resize224 + HorizontalFlip","cutmix":False,"unique_trainings":36,"groups":groups,"rows":rows,"errors":errors}
    out=a.root/"summary/aircraft_rded_random_fullframe.json";out.parent.mkdir(parents=True,exist_ok=True)
    t=out.with_suffix(".json.tmp");t.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");os.replace(t,out)
    print(json.dumps({"status":payload["status"],"paired_rows":len(rows),"errors":errors},indent=2))
    if payload["status"]!="complete":raise SystemExit(1)
if __name__=="__main__":main()
