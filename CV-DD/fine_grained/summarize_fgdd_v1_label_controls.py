import argparse,json,os,statistics
from pathlib import Path
def load(p):return json.loads(p.read_text())
def stats(v):v=list(map(float,v));return {'mean':statistics.mean(v),'sample_std':statistics.stdev(v),'values':v}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--fgdd-root',required=True,type=Path);p.add_argument('--control-root',required=True,type=Path);a=p.parse_args();rows=[];errors=[]
 for condition in ('view_soft_cutmix','hard_fullframe','fixed_image_soft_cutmix'):
  for method in ('r0_random','r3_fg_confusion_gain'):
   for student in (42,43,44):
    if condition=='view_soft_cutmix':path=(a.control_root/f'results/fullframe_cutmix/random_real/source_seed0/ipc3_sseed{student}.json' if method=='r0_random' else a.fgdd_root/f'results/r3_fg_confusion_gain/ipc3_sseed{student}.json')
    else:path=a.root/f'results/{condition}/{method}/ipc3_sseed{student}.json'
    try:
     x=load(path);target=x['training_target'];assert x['student_seed']==student and x['batch_size']==20 and x['gradient_accumulation_steps']==2 and x['epochs']==400
     if condition=='hard_fullframe':assert target=='fkd_replay_hard_cutmix_ce' and x['mix_type'] is None
     elif condition=='fixed_image_soft_cutmix':assert target=='fkd_probability_target' and x['mix_type']=='cutmix' and x['fkd_target_probabilities'] is True
     rows.append({'condition':condition,'method':method,'student_seed':student,'best':x['best_top1'],'final':x['final_epoch_top1'],'best_epoch':x['best_epoch'],'initial_hash':x['initial_model_sha256']})
    except Exception as e:errors.append({'path':str(path),'error':repr(e)})
 groups=[]
 for condition in ('view_soft_cutmix','hard_fullframe','fixed_image_soft_cutmix'):
  z=[r for r in rows if r['condition']==condition];r0={r['student_seed']:r for r in z if r['method']=='r0_random'};r3=[r for r in z if r['method']=='r3_fg_confusion_gain']
  groups.append({'condition':condition,'r0_best':stats(r['best'] for r in r0.values()),'r0_final':stats(r['final'] for r in r0.values()),'r3_best':stats(r['best'] for r in r3),'r3_final':stats(r['final'] for r in r3),'r3_minus_r0_best':stats(r['best']-r0[r['student_seed']]['best'] for r in r3),'r3_minus_r0_final':stats(r['final']-r0[r['student_seed']]['final'] for r in r3),'r0_best_epochs':[r['best_epoch'] for r in r0.values()],'r3_best_epochs':[r['best_epoch'] for r in r3]})
 for student in (42,43,44):
  hashes={r['initial_hash'] for r in rows if r['student_seed']==student}
  if len(hashes)!=1:errors.append({'student_seed':student,'error':'initial hash mismatch'})
 try:context=load(a.root/'fixed/fixed_label_context.json');manifests={m:load(a.root/f'derived_fkd/{m}/ipc3_bs20_ipc3/fixed_target_manifest.json') for m in ('r0_random','r3_fg_confusion_gain')}
 except Exception as e:context={};manifests={};errors.append({'fixed_label_audit':repr(e)})
 result={'status':'complete' if len(rows)==18 and not errors else 'failed','dataset':'A_imsize224','ipc':3,'new_student_runs':12,'groups':groups,'rows':rows,'fixed_label_context':context,'derived_manifests':manifests,'errors':errors};out=a.root/'summary/fgdd_v1_label_controls.json';out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');os.replace(tmp,out);print(json.dumps({'status':result['status'],'rows':len(rows),'errors':errors},indent=2));
 if result['status']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
