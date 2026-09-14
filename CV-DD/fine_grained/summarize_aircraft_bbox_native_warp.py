import argparse,json,os,statistics
from pathlib import Path
def load(p):return json.loads(p.read_text())
def stats(v):v=list(map(float,v));return {'mean':statistics.mean(v),'sample_std':statistics.stdev(v),'values':v}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--r0-root',required=True,type=Path);p.add_argument('--letterbox-root',required=True,type=Path);a=p.parse_args();rows=[];errors=[]
 for method in ('r0_random','native_subject_letterbox','native_subject_warp'):
  for s in (42,43,44):
   if method=='r0_random':path=a.r0_root/f'results/fullframe_cutmix/random_real/source_seed0/ipc3_sseed{s}.json'
   elif method=='native_subject_letterbox':path=a.letterbox_root/f'results/native_subject/ipc3_sseed{s}.json'
   else:path=a.root/f'results/native_subject_warp/ipc3_sseed{s}.json'
   try:x=load(path);assert x['training_target']=='fkd_soft_label' and x['mix_type']=='cutmix' and x['student_seed']==s;rows.append({'method':method,'student_seed':s,'best':x['best_top1'],'final':x['final_epoch_top1'],'best_epoch':x['best_epoch'],'initial_hash':x['initial_model_sha256']})
   except Exception as e:errors.append({'path':str(path),'error':repr(e)})
 by={m:{r['student_seed']:r for r in rows if r['method']==m} for m in ('r0_random','native_subject_letterbox','native_subject_warp')};groups=[]
 for method,zmap in by.items():
  z=list(zmap.values());groups.append({'method':method,'best':stats(r['best'] for r in z),'final':stats(r['final'] for r in z),'best_epochs':[r['best_epoch'] for r in z],'delta_vs_r0_best':stats(r['best']-by['r0_random'][r['student_seed']]['best'] for r in z),'delta_vs_r0_final':stats(r['final']-by['r0_random'][r['student_seed']]['final'] for r in z)})
 contrasts={}
 for left,right in (('native_subject_warp','native_subject_letterbox'),('native_subject_warp','r0_random')):
  for metric in ('best','final'):contrasts[f'{left}_minus_{right}_{metric}']=stats(by[left][s][metric]-by[right][s][metric] for s in (42,43,44))
 for s in (42,43,44):
  if len({by[m][s]['initial_hash'] for m in by})!=1:errors.append({'student_seed':s,'error':'hash mismatch'})
 try:construction=load(a.root/'construction/construction_manifest.json');alignment=load(a.root/'audits/fkd_vs_r0_metadata.json');assert alignment['status']=='complete'
 except Exception as e:construction={};alignment={};errors.append({'audit':repr(e)})
 result={'status':'complete' if len(rows)==9 and not errors else 'failed','dataset':'A_imsize224','ipc':3,'condition':'fullframe+flip+CutMix actual-view Teacher T20 KL','new_fkd':1,'new_students':3,'groups':groups,'contrasts':contrasts,'construction_metrics':construction.get('metrics'),'alignment':alignment,'rows':rows,'errors':errors};out=a.root/'summary/aircraft_bbox_native_warp.json';out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');os.replace(tmp,out);print(json.dumps({'status':result['status'],'rows':len(rows),'errors':errors},indent=2));
 if result['status']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
