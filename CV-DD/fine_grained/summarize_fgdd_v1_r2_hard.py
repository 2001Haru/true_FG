import argparse,json,os,statistics
from pathlib import Path
def load(p):return json.loads(p.read_text())
def stats(v):v=list(map(float,v));return {'mean':statistics.mean(v),'sample_std':statistics.stdev(v),'values':v}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();rows=[];errors=[]
 for method in ('r0_random','r2_ordinary_gain','r3_fg_confusion_gain'):
  for student in (42,43,44):
   path=a.root/f'results/hard_fullframe/{method}/ipc3_sseed{student}.json'
   try:
    x=load(path);assert x['training_target']=='fkd_replay_hard_cutmix_ce' and x['mix_type'] is None and x['student_seed']==student and x['student_protocol_name']=='standard_protocol_v2_fgdd_hard_fullframe';rows.append({'method':method,'student_seed':student,'best':x['best_top1'],'final':x['final_epoch_top1'],'best_epoch':x['best_epoch'],'initial_hash':x['initial_model_sha256']})
   except Exception as e:errors.append({'path':str(path),'error':repr(e)})
 by={m:{r['student_seed']:r for r in rows if r['method']==m} for m in ('r0_random','r2_ordinary_gain','r3_fg_confusion_gain')};groups=[]
 for method in by:
  z=list(by[method].values());groups.append({'method':method,'best':stats(r['best'] for r in z),'final':stats(r['final'] for r in z),'best_epochs':[r['best_epoch'] for r in z]})
 contrasts={}
 for left,right in (('r2_ordinary_gain','r0_random'),('r3_fg_confusion_gain','r2_ordinary_gain'),('r3_fg_confusion_gain','r0_random')):
  contrasts[f'{left}_minus_{right}_best']=stats(by[left][s]['best']-by[right][s]['best'] for s in (42,43,44));contrasts[f'{left}_minus_{right}_final']=stats(by[left][s]['final']-by[right][s]['final'] for s in (42,43,44))
 for s in (42,43,44):
  if len({by[m][s]['initial_hash'] for m in by})!=1:errors.append({'student_seed':s,'error':'initial hash mismatch'})
 result={'status':'complete' if len(rows)==9 and not errors else 'failed','dataset':'A_imsize224','ipc':3,'condition':'fullframe_flip_hard_T1_CE','new_student_runs':3,'groups':groups,'contrasts':contrasts,'rows':rows,'errors':errors};out=a.root/'summary/fgdd_v1_r2_hard.json';tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');os.replace(tmp,out);print(json.dumps({'status':result['status'],'rows':len(rows),'errors':errors},indent=2));
 if result['status']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
