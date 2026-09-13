"""Summarize paired Aircraft IPC3 RDED tile-vs-full-mosaic Soft results."""
import argparse,json,os,statistics
from pathlib import Path
def stats(v):v=list(map(float,v));return {'mean':statistics.mean(v),'sample_std':statistics.stdev(v),'values':v}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();rows=[];errors=[]
 for g in (42,43,44):
  for s in (42,43,44):
   path=a.root/f'results/gseed{g}/ipc3_sseed{s}.json'
   try:
    x=json.loads(path.read_text());m=x['rded_tile_soft_v2'];c=json.loads(Path(m['control_result']).read_text());rows.append({'generation_seed':g,'student_seed':s,'tile_best':x['best_top1'],'tile_final':x['final_epoch_top1'],'control_best':c['best_top1'],'control_final':c['final_epoch_top1'],'tile_minus_control_best':x['best_top1']-c['best_top1'],'tile_minus_control_final':x['final_epoch_top1']-c['final_epoch_top1'],'tile_best_epoch':x['best_epoch'],'hash':x['initial_model_sha256']})
   except Exception as e:errors.append({'path':str(path),'error':str(e)})
 payload={'status':'complete' if len(rows)==9 and not errors else 'failed','dataset':'A_imsize224','ipc':3,'method':'RDED-tile','generation_seeds':[42,43,44],'student_seeds':[42,43,44],**{k:stats(r[k] for r in rows) for k in ('tile_best','tile_final','control_best','control_final','tile_minus_control_best','tile_minus_control_final')},'rows':rows,'errors':errors}
 out=a.root/'summary/rded_tile_soft_v2.json';out.parent.mkdir(parents=True,exist_ok=True);t=out.with_suffix('.json.tmp');t.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');os.replace(t,out);print(json.dumps({'status':payload['status'],'rows':len(rows),'errors':errors},indent=2))
 if payload['status']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
