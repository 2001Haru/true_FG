"""Audit dynamic two-strip feasibility and per-class density-gap attribution."""
import argparse,json,math,statistics
from pathlib import Path
import numpy as np
def stats(v):v=list(map(float,v));return {'mean':statistics.mean(v),'median':statistics.median(v),'min':min(v),'max':max(v)}
def ranks(x):
 order=np.argsort(x,kind='mergesort');out=np.empty(len(x),float);i=0
 while i<len(x):
  j=i+1
  while j<len(x) and x[order[j]]==x[order[i]]:j+=1
  out[order[i:j]]=(i+j-1)/2+1;i=j
 return out
def spearman(x,y):return float(np.corrcoef(ranks(np.asarray(x)),ranks(np.asarray(y)))[0,1])
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifests',required=True,type=Path,nargs=3);p.add_argument('--reference-results',required=True,type=Path);p.add_argument('--bbox-results',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();feas={};allrows=[]
 for r,path in enumerate(a.manifests):
  x=json.loads(path.read_text());rows=[]
  for parent in x['records']:
   heights=[s['bbox_rows'] for s in parent['sources']];total=sum(heights);required_floor=total+.1*(448-total);equal_density=min(1.,(224-.1*(448-total))/total)
   rows.append({'class_id':parent['class_id'],'slot':parent['slot'],'heights':heights,'height_sum':total,'feasible_no_background_floor':total<=224,'required_with_background_floor':required_floor,'feasible_background_floor_0p1':required_floor<=224,'max_equal_subject_density_with_floor':equal_density,'slack_no_floor':224-total,'slack_with_floor':224-required_floor})
  allrows+=rows;feas[str(r)]={'slots':len(rows),'subject_density1_feasible_no_floor':sum(z['feasible_no_background_floor'] for z in rows),'subject_density1_feasible_floor0p1':sum(z['feasible_background_floor_0p1'] for z in rows),'height_sum':stats(z['height_sum'] for z in rows),'slack_no_floor':stats(z['slack_no_floor'] for z in rows),'slack_floor0p1':stats(z['slack_with_floor'] for z in rows),'max_equal_subject_density_floor0p1':stats(z['max_equal_subject_density_with_floor'] for z in rows)}
 feas['all900']={'slots':len(allrows),'subject_density1_feasible_no_floor':sum(z['feasible_no_background_floor'] for z in allrows),'subject_density1_feasible_floor0p1':sum(z['feasible_background_floor_0p1'] for z in allrows),'height_sum':stats(z['height_sum'] for z in allrows),'slack_no_floor':stats(z['slack_no_floor'] for z in allrows),'slack_floor0p1':stats(z['slack_with_floor'] for z in allrows),'max_equal_subject_density_floor0p1':stats(z['max_equal_subject_density_with_floor'] for z in allrows)}
 manifest=json.loads(a.manifests[0].read_text());gaps={}
 for c in range(100):
  sources=[s for row in manifest['records'] if row['class_id']==c for s in row['sources']];num=sum(s['bbox_rows']*(1-max(s['density'])) for s in sources);den=sum(s['bbox_rows'] for s in sources);gaps[c]=num/den
 ref=[];bbox=[]
 for seed in (42,43,44):
  ref.append({z['class_id']:z['accuracy'] for z in json.loads((a.reference_results/f'sseed{seed}.json').read_text())['per_class']});bbox.append({z['class_id']:z['accuracy'] for z in json.loads((a.bbox_results/f'sseed{seed}.json').read_text())['per_class']})
 gap=[gaps[c] for c in range(100)];decline=[statistics.mean(ref[k][c]-bbox[k][c] for k in range(3)) for c in range(100)];rho=spearman(gap,decline);per=[spearman(gap,[ref[k][c]-bbox[k][c] for c in range(100)]) for k in range(3)]
 rng=np.random.default_rng(20260916);boot=[]
 for _ in range(10000):
  idx=rng.integers(0,100,100);boot.append(spearman(np.asarray(gap)[idx],np.asarray(decline)[idx]))
 result={'status':'complete','feasibility':feas,'definitions':{'no_floor':'density1 requires h1+h2<=224','floor0p1':'density1 subjects plus background density0.1 requires h1+h2+0.1*(448-h1-h2)<=224'},'per_class_attribution':{'accuracy':'best-checkpoint per-class official-test accuracy averaged over Student42/43/44','density_gap':'bbox-height-weighted mean of 1-subject_density over six class sources','spearman_rho':rho,'student_rhos':per,'bootstrap95':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'classes':100,'rows':[{'class_id':c,'density_gap':gap[c],'accuracy_decline_points':decline[c]} for c in range(100)]}}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({**result,'per_class_attribution':{k:v for k,v in result['per_class_attribution'].items() if k!='rows'}},indent=2))
if __name__=='__main__':main()
