"""Summarize Soft intermediate cells and the full label × RRC × CutMix cube."""
import argparse,json,os,statistics
from pathlib import Path
def stats(v):v=list(map(float,v));return {'mean':statistics.mean(v),'sample_std':statistics.stdev(v),'values':v}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--offoff-root',required=True,type=Path);p.add_argument('--onon-root',required=True,type=Path);p.add_argument('--hard-middle-root',required=True,type=Path);a=p.parse_args();rows=[];errors=[]
 for arm in ('rrc_no_cutmix','fullframe_cutmix'):
  for method,seeds in (('random_real',(0,1,2)),('original_rded',(42,43,44))):
   for q in seeds:
    for s in (42,43,44):
     path=a.root/f'results/{arm}/{method}/source_seed{q}/ipc3_sseed{s}.json'
     try:
      x=json.loads(path.read_text());rows.append({'arm':arm,'method':method,'source_seed':q,'student_seed':s,'best':x['best_top1'],'final':x['final_epoch_top1'],'best_epoch':x['best_epoch'],'hash':x['initial_model_sha256']})
     except Exception as e:errors.append({'path':str(path),'error':str(e)})
 groups=[]
 for arm in ('rrc_no_cutmix','fullframe_cutmix'):
  for method in ('random_real','original_rded'):
   z=[r for r in rows if r['arm']==arm and r['method']==method];groups.append({'arm':arm,'method':method,'count':len(z),'best':stats(r['best'] for r in z),'final':stats(r['final'] for r in z),'best_epochs':[r['best_epoch'] for r in z]})
 off=json.loads((a.offoff_root/'summary/aircraft_rded_random_fullframe.json').read_text());on=json.loads((a.onon_root/'summary/aircraft_hard_replay_v2.json').read_text());hard=json.loads((a.hard_middle_root/'summary/aircraft_hard_aug_factorial.json').read_text())
 endpoints={}
 for method in ('random_real','original_rded'):
  g0=next(g for g in off['groups'] if g['method']==method);g1=next(g for g in on['groups'] if g['method']==method and g['ipc']==3)
  endpoints[method]={'fullframe_no_cutmix':{'best':g0['soft_best'],'final':g0['soft_final']},'rrc_cutmix':{'best':g1['soft_best'],'final':g1['soft_final']}}
 hashes={}
 for s in (42,43,44):
  u=sorted({r['hash'] for r in rows if r['student_seed']==s});hashes[str(s)]={'unique':len(u),'hashes':u}
  if rows and len(u)!=1:errors.append({'student_seed':s,'error':'hash mismatch'})
 payload={'status':'complete' if len(rows)==36 and not errors else 'failed','dataset':'A_imsize224','ipc':3,'label':'soft','new_results':36,'groups':groups,'endpoints':endpoints,'hard_factorial_summary':str((a.hard_middle_root/'summary/aircraft_hard_aug_factorial.json').resolve()),'rows':rows,'initial_hash_audit':hashes,'errors':errors}
 out=a.root/'summary/aircraft_soft_aug_factorial.json';out.parent.mkdir(parents=True,exist_ok=True);t=out.with_suffix('.json.tmp');t.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');os.replace(t,out);print(json.dumps({'status':payload['status'],'rows':len(rows),'errors':errors},indent=2))
 if payload['status']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
