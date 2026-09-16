"""Summarize six-source Soft reference/BBox over three source selections."""
import argparse,json,statistics
from pathlib import Path
S=(42,43,44)
def st(v):v=list(map(float,v));return {'mean':statistics.mean(v),'sample_sd':statistics.stdev(v),'values':v}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--parent',required=True,type=Path);a=p.parse_args();rows={};errors=[]
 for r in (0,1,2):
  rows[r]={'reference':{},'compressed':{}}
  for mode in ('reference','compressed'):
   for s in S:
    path=(a.parent/f'results/soft/{mode}/sseed{s}.json') if r==0 else (a.root/f'results/rseed{r}/{mode}/sseed{s}.json')
    try:rows[r][mode][s]=json.loads(path.read_text())
    except Exception as e:errors.append({'path':str(path),'error':repr(e)})
 by={}
 for r in (0,1,2):
  by[str(r)]={m:{'best':st(rows[r][m][s]['best_top1'] for s in S),'final':st(rows[r][m][s]['final_epoch_top1'] for s in S)} for m in ('reference','compressed')}
  by[str(r)]['compressed_minus_reference']={'best':st(rows[r]['compressed'][s]['best_top1']-rows[r]['reference'][s]['best_top1'] for s in S),'final':st(rows[r]['compressed'][s]['final_epoch_top1']-rows[r]['reference'][s]['final_epoch_top1'] for s in S)}
 refs=[rows[r]['reference'][s] for r in (0,1,2) for s in S];comps=[rows[r]['compressed'][s] for r in (0,1,2) for s in S]
 aggregate={'reference':{'best':st(x['best_top1'] for x in refs),'final':st(x['final_epoch_top1'] for x in refs)},'compressed':{'best':st(x['best_top1'] for x in comps),'final':st(x['final_epoch_top1'] for x in comps)},'compressed_minus_reference':{'best':st(y['best_top1']-x['best_top1'] for x,y in zip(refs,comps)),'final':st(y['final_epoch_top1']-x['final_epoch_top1'] for x,y in zip(refs,comps))}}
 result={'status':'complete' if not errors else 'failed','protocol':'aircraft_six_source_soft_source_seed_extension_v1','by_source_seed':by,'all9':aggregate,'three_source_r0_all9_descriptive_reference':{'best':74.2574,'final':73.8374},'new_fkd_sets':4,'new_student_runs':12,'errors':errors}
 out=a.root/'summary/summary.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));
 if result['status']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
