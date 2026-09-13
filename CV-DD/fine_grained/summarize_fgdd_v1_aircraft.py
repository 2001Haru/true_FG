import argparse,json,os,statistics
from pathlib import Path

def stats(values):
    values=list(map(float,values));return {'mean':statistics.mean(values),'sample_std':statistics.stdev(values),'values':values}
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--control-root',required=True,type=Path);a=p.parse_args();errors=[];rows=[]
    methods=('r0_random','r1_teacher_confidence','r2_ordinary_gain','r3_fg_confusion_gain')
    for method in methods:
      for student in (42,43,44):
       path=(a.control_root/f'results/fullframe_cutmix/random_real/source_seed0/ipc3_sseed{student}.json' if method=='r0_random' else a.root/f'results/{method}/ipc3_sseed{student}.json')
       try:
        x=json.loads(path.read_text());
        if method!='r0_random': assert x['fgdd_v1']['method']==method
        rows.append({'method':method,'student_seed':student,'best':x['best_top1'],'final':x['final_epoch_top1'],'best_epoch':x['best_epoch'],'initial_hash':x['initial_model_sha256'],'path':str(path)})
       except Exception as exc:errors.append({'path':str(path),'error':repr(exc)})
    groups=[]
    control={r['student_seed']:r for r in rows if r['method']=='r0_random'}
    for method in methods:
      z=[r for r in rows if r['method']==method];g={'method':method,'count':len(z),'best':stats(r['best'] for r in z),'final':stats(r['final'] for r in z),'best_epochs':[r['best_epoch'] for r in z]}
      if method!='r0_random':g['delta_vs_r0_best']=stats(r['best']-control[r['student_seed']]['best'] for r in z);g['delta_vs_r0_final']=stats(r['final']-control[r['student_seed']]['final'] for r in z)
      groups.append(g)
    for student in (42,43,44):
      hashes={r['initial_hash'] for r in rows if r['student_seed']==student}
      if len(hashes)!=1:errors.append({'student_seed':student,'error':'initial hash mismatch'})
    manifest={}
    try:manifest=json.loads((a.root/'selection/selection_manifest.json').read_text());assert manifest['status']=='complete'
    except Exception as exc:errors.append({'manifest_error':repr(exc)})
    result={'status':'complete' if len(rows)==12 and not errors else 'failed','dataset':'A_imsize224','ipc':3,'student_seeds':[42,43,44],'new_student_runs':9,'groups':groups,'rows':rows,'selection_summary':{k:manifest.get(k) for k in ('anchors','candidate_pool','scorer','teacher_scoring_context','neighbor_rule','reference_weights','third_choice_overlap_classes','teacher_scoring_queries')},'errors':errors}
    out=a.root/'summary/fgdd_v1_aircraft.json';out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');os.replace(tmp,out);print(json.dumps({'status':result['status'],'rows':len(rows),'errors':errors},indent=2));
    if result['status']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
