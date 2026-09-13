import argparse,json,os,statistics
from pathlib import Path

METHODS=(("random_real",(0,1,2)),("original_rded",(42,43,44)))
STUDENTS=(42,43,44)
def load(path): return json.loads(path.read_text())
def stats(values):
    values=list(map(float,values));return {"mean":statistics.mean(values),"sample_std":statistics.stdev(values),"values":values}
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--old-teacher-hard-root',type=Path,required=True);p.add_argument('--soft-root',type=Path,required=True);p.add_argument('--hard-root',type=Path,required=True);a=p.parse_args();rows=[];errors=[]
    for method,sources in METHODS:
      for source in sources:
       for student in STUDENTS:
        try:
         old=(method=='random_real' and source==0)or(method=='original_rded' and source==42)
         rrc_th=load((a.old_teacher_hard_root if old else a.root)/(f"results/{method}/source_seed{source}/ipc3_sseed{student}.json" if old else f"results/teacher_hard_rrc/{method}/source_seed{source}/ipc3_sseed{student}.json"))
         full_th=load(a.root/f"results/teacher_hard_fullframe/{method}/source_seed{source}/ipc3_sseed{student}.json")
         perm=load(a.root/f"results/preserved_permutation/{method}/source_seed{source}/ipc3_sseed{student}.json")
         hard_rrc=load(a.hard_root/f"results/rrc_no_cutmix/{method}/source_seed{source}/ipc3_sseed{student}.json")
         soft_rrc=load(a.soft_root/f"results/rrc_no_cutmix/{method}/source_seed{source}/ipc3_sseed{student}.json")
         full=load(Path('/linxi/dataset/FG_RDED_Random_fullframe_v2/aircraft_ipc3_v1')/f"results/hard/{method}/source_seed{source}/ipc3_sseed{student}.json")
         soft_full=load(Path('/linxi/dataset/FG_RDED_Random_fullframe_v2/aircraft_ipc3_v1')/f"results/soft/{method}/source_seed{source}/ipc3_sseed{student}.json")
         assert rrc_th['training_target']==full_th['training_target']=='fkd_teacher_argmax_ce';assert perm['training_target']=='fkd_soft_label'
         hashes={x['initial_model_sha256'] for x in (rrc_th,full_th,perm,hard_rrc,soft_rrc,full,soft_full)};assert len(hashes)==1
         rows.append({'method':method,'source_seed':source,'student_seed':student,'rrc_teacher_hard_best':rrc_th['best_top1'],'rrc_teacher_hard_final':rrc_th['final_epoch_top1'],'full_teacher_hard_best':full_th['best_top1'],'full_teacher_hard_final':full_th['final_epoch_top1'],'rrc_hard_best':hard_rrc['best_top1'],'rrc_hard_final':hard_rrc['final_epoch_top1'],'full_hard_best':full['best_top1'],'full_hard_final':full['final_epoch_top1'],'rrc_soft_best':soft_rrc['best_top1'],'rrc_soft_final':soft_rrc['final_epoch_top1'],'full_soft_best':soft_full['best_top1'],'full_soft_final':soft_full['final_epoch_top1'],'perm_best':perm['best_top1'],'perm_final':perm['final_epoch_top1']})
        except Exception as exc:errors.append({'method':method,'source':source,'student':student,'error':repr(exc)})
    groups=[]
    for method,_ in METHODS:
      z=[r for r in rows if r['method']==method];g={'method':method,'count':len(z)}
      for metric in ('rrc_teacher_hard_best','rrc_teacher_hard_final','full_teacher_hard_best','full_teacher_hard_final','rrc_hard_best','rrc_hard_final','full_hard_best','full_hard_final','rrc_soft_best','rrc_soft_final','full_soft_best','full_soft_final','perm_best','perm_final'):
       g[metric]=stats(r[metric] for r in z)
      for suffix in ('best','final'):
       g[f'rrc_teacher_hard_gain_{suffix}']=stats(r[f'rrc_teacher_hard_{suffix}']-r[f'rrc_hard_{suffix}'] for r in z)
       g[f'full_teacher_hard_gain_{suffix}']=stats(r[f'full_teacher_hard_{suffix}']-r[f'full_hard_{suffix}'] for r in z)
       g[f'rrc_specific_gain_{suffix}']=stats((r[f'rrc_teacher_hard_{suffix}']-r[f'rrc_hard_{suffix}'])-(r[f'full_teacher_hard_{suffix}']-r[f'full_hard_{suffix}']) for r in z)
       g[f'permutation_minus_soft_{suffix}']=stats(r[f'perm_{suffix}']-r[f'rrc_soft_{suffix}'] for r in z)
      groups.append(g)
    interactions={}
    for suffix in ('best','final'):
      random=next(g for g in groups if g['method']=='random_real');rded=next(g for g in groups if g['method']=='original_rded')
      interactions[f'permutation_relative_source_interaction_{suffix}']=rded[f'permutation_minus_soft_{suffix}']['mean']-random[f'permutation_minus_soft_{suffix}']['mean']
    manifests=[]
    for method,sources in METHODS:
     for source in sources:
      try:manifests.append(load(a.root/f"permuted_fkd/{method}/source_seed{source}/ipc3_bs20_ipc3/permutation_manifest.json"))
      except Exception as exc:errors.append({'permutation_manifest':f'{method}/{source}','error':repr(exc)})
    result={'status':'complete' if len(rows)==18 and len(manifests)==6 and not errors else 'failed','dataset':'A_imsize224','ipc':3,'new_student_runs':48,'teacher_queries':0,'rows':rows,'groups':groups,'interactions':interactions,'permutation_manifests':manifests,'errors':errors}
    out=a.root/'summary/aircraft_teacher_hard_controls.json';tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');os.replace(tmp,out);print(json.dumps({'status':result['status'],'rows':len(rows),'errors':errors},indent=2));
    if result['status']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
