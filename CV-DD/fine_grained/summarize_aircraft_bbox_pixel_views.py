import argparse,json,os,statistics
from pathlib import Path
def load(p):return json.loads(p.read_text())
def stats(v):v=list(map(float,v));return {'mean':statistics.mean(v),'sample_std':statistics.stdev(v),'values':v}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--control-root',required=True,type=Path);a=p.parse_args();rows=[];errors=[]
 for method in ('r0_random','native_subject','lowres_subject'):
  for student in (42,43,44):
   path=a.control_root/f'results/fullframe_cutmix/random_real/source_seed0/ipc3_sseed{student}.json' if method=='r0_random' else a.root/f'results/{method}/ipc3_sseed{student}.json'
   try:x=load(path);assert x['training_target']=='fkd_soft_label' and x['mix_type']=='cutmix' and x['student_seed']==student;rows.append({'method':method,'student_seed':student,'best':x['best_top1'],'final':x['final_epoch_top1'],'best_epoch':x['best_epoch'],'initial_hash':x['initial_model_sha256']})
   except Exception as e:errors.append({'path':str(path),'error':repr(e)})
 by={m:{r['student_seed']:r for r in rows if r['method']==m} for m in ('r0_random','native_subject','lowres_subject')};groups=[]
 for method,zmap in by.items():
  z=list(zmap.values());groups.append({'method':method,'best':stats(r['best'] for r in z),'final':stats(r['final'] for r in z),'best_epochs':[r['best_epoch'] for r in z],'delta_vs_r0_best':stats(r['best']-by['r0_random'][r['student_seed']]['best'] for r in z),'delta_vs_r0_final':stats(r['final']-by['r0_random'][r['student_seed']]['final'] for r in z)})
 contrasts={}
 for left,right in (('native_subject','lowres_subject'),('native_subject','r0_random'),('lowres_subject','r0_random')):
  for metric in ('best','final'):contrasts[f'{left}_minus_{right}_{metric}']=stats(by[left][s][metric]-by[right][s][metric] for s in (42,43,44))
 for s in (42,43,44):
  if len({by[m][s]['initial_hash'] for m in by})!=1:errors.append({'student_seed':s,'error':'initial hash mismatch'})
 try:construction=load(a.root/'construction/construction_manifest.json');assert construction['training_gate']['passed'];fkds={m:load(a.root/f'fkd/{m}/ipc3_bs20_ipc3/fkd_audit.json') for m in ('native_subject','lowres_subject')}
 except Exception as e:construction={};fkds={};errors.append({'audit':repr(e)})
 result={'status':'complete' if len(rows)==9 and not errors else 'failed','dataset':'A_imsize224','ipc':3,'new_fkd':2,'new_student_runs':6,'groups':groups,'contrasts':contrasts,'construction_summary':{k:construction.get(k) for k in ('margin_fraction_each_bbox_side','cropped_fraction','window_area_fraction','native_letterbox_content_fraction','lowres_letterbox_content_fraction','r0_vs_raw_direct_resize_pixel_mae','output_rule','lowres_rule')},'rows':rows,'errors':errors};out=a.root/'summary/aircraft_bbox_pixel_views.json';out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');os.replace(tmp,out);print(json.dumps({'status':result['status'],'rows':len(rows),'errors':errors},indent=2));
 if result['status']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
