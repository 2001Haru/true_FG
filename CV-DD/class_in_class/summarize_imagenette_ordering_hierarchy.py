"""Summarize O/B/A/Aprime with manifest as the independent unit."""

import argparse,json,math,os,statistics
from pathlib import Path
from scipy.stats import t

PAIRS={3:(42,43),4:(43,44),5:(42,44),6:(42,43),7:(43,44),8:(42,44),9:(42,43),10:(43,44),11:(42,44)}
ARMS=("O","B","A","Aprime")

def stats(v):
 v=list(map(float,v));return {"mean":statistics.mean(v),"sample_std":statistics.stdev(v),"values":v,"positive":sum(x>0 for x in v),"zero":sum(x==0 for x in v),"negative":sum(x<0 for x in v)}
def block_adjusted(rows,key):
 values=[float(row["contrasts"][key]) for row in rows];mean=statistics.mean(values);blocks={}
 for row,value in zip(rows,values):blocks.setdefault(tuple(row["student_pair"]),[]).append(value)
 block_means={block:statistics.mean(group) for block,group in blocks.items()}
 residual=[value-block_means[tuple(row["student_pair"])] for row,value in zip(rows,values)]
 df=len(values)-len(blocks);mse=sum(value*value for value in residual)/df;se=math.sqrt(mse/len(values));critical=float(t.ppf(.975,df));tv=mean/se if se else float('inf');p=float(2*t.sf(abs(tv),df))
 return {"mean":mean,"student_pair_blocks":{"-".join(map(str,k)):v for k,v in block_means.items()},"residual_mse":mse,"standard_error":se,"df":df,"ci95":[mean-critical*se,mean+critical*se],"two_sided_p":p}
def holm_two(pvalues):
 order=sorted(range(len(pvalues)),key=lambda i:pvalues[i]);adjusted=[0.0]*len(pvalues);running=0.0
 for rank,index in enumerate(order):running=max(running,(len(pvalues)-rank)*pvalues[index]);adjusted[index]=min(running,1.0)
 return adjusted
def atomic(path,p):
 path.parent.mkdir(parents=True,exist_ok=True);t=path.with_suffix('.tmp');t.write_text(json.dumps(p,indent=2,sort_keys=True)+'\n');os.replace(t,path)

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--experiment-root',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();root=a.experiment_root.resolve();errors=[];tuples=[]
 for r,seeds in PAIRS.items():
  for s in seeds:
   row={"manifest_seed":r,"student_seed":s,"arms":{}}
   hashes=set()
   for arm in ARMS:
    path=root/f"results/rseed{r}/sseed{s}/{arm}.json"
    if not path.is_file():errors.append(f"missing {path}");continue
    p=json.load(open(path));row["arms"][arm]={"nll":p["final_loss"],"top1":p["final_top1"],"path":str(path.resolve())};hashes.add((p["train_manifest_sha256"],p["initial_student_state_sha256"],p["ordering_templates_sha256"]))
   if len(row["arms"])==4:
    if len(hashes)!=1:errors.append(f"pairing mismatch r{r}s{s}")
    L={k:v["nll"] for k,v in row["arms"].items()};T={k:v["top1"] for k,v in row["arms"].items()}
    row["contrasts"]={"delta_control_nll":L["Aprime"]-L["A"],"delta_view_nll":L["B"]-L["O"],"delta_image_nll":L["A"]-L["B"],"delta_control_top1":T["A"]-T["Aprime"],"delta_view_top1":T["O"]-T["B"],"delta_image_top1":T["B"]-T["A"]}
    tuples.append(row)
 manifest_rows=[]
 for r,seeds in PAIRS.items():
  rs=[x for x in tuples if x["manifest_seed"]==r]
  if len(rs)!=2:continue
  manifest_rows.append({"manifest_seed":r,"student_pair":list(seeds),"arm_nll":{arm:statistics.mean(x["arms"][arm]["nll"] for x in rs) for arm in ARMS},"arm_top1":{arm:statistics.mean(x["arms"][arm]["top1"] for x in rs) for arm in ARMS},"contrasts":{key:statistics.mean(x["contrasts"][key] for x in rs) for key in rs[0]["contrasts"]}})
 summary={key:stats(row["contrasts"][key] for row in manifest_rows) for key in manifest_rows[0]["contrasts"]} if len(manifest_rows)==9 else {}
 block_summary={key:block_adjusted(manifest_rows,key) for key in manifest_rows[0]["contrasts"]} if len(manifest_rows)==9 else {}
 if block_summary:
  secondary=["delta_view_nll","delta_image_nll"];adjusted=holm_two([block_summary[key]["two_sided_p"] for key in secondary])
  for key,value in zip(secondary,adjusted):block_summary[key]["holm_adjusted_p_across_two_secondary_nll_tests"]=value
 arm_summary={metric:{arm:stats(row[f"arm_{metric}"][arm] for row in manifest_rows) for arm in ARMS} for metric in ("nll","top1")} if len(manifest_rows)==9 else {}
 payload={"status":"complete" if len(tuples)==18 and len(manifest_rows)==9 and not errors else "incomplete","expected_results":72,"found_results":len(tuples)*4,"errors":errors,"independent_unit":"nine manifests; average two Students within manifest; Student pair is fixed block","primary":"delta_control_nll=L_Aprime-L_A","secondary":["delta_view_nll=L_B-L_O","delta_image_nll=L_A-L_B"],"arm_summary_across_manifests":arm_summary,"contrast_summary_across_manifests":summary,"fixed_student_pair_block_inference":block_summary,"manifest_rows":manifest_rows,"tuples":tuples}
 atomic(a.output,payload);print(json.dumps({k:payload[k] for k in ('status','expected_results','found_results','errors')},indent=2));
 if payload['status']!='complete':raise RuntimeError('incomplete ordering matrix')
if __name__=='__main__':main()
