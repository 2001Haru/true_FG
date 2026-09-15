"""Summarize Aircraft IPC3 DINO selections under Soft-v2 with RRC off/on."""
import argparse,json,statistics
from pathlib import Path

MODES=("fullframe_cutmix","rrc_cutmix");SEEDS=(42,43,44);METRICS=("best_top1","final_epoch_top1")
KARMS=("spherical_kmeans3_rseed0","spherical_kmeans3_rseed1")

def stats(v):return {"mean":statistics.mean(v),"sample_sd":statistics.stdev(v),"values":v}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);p.add_argument('--r0-full',required=True,type=Path);p.add_argument('--r0-rrc',required=True,type=Path);a=p.parse_args()
 rows={mode:{} for mode in MODES};r0={}
 for mode in MODES:
  rows[mode]['global_center_top3']={s:json.loads((a.root/f'results/{mode}/global_center_top3/ipc3_sseed{s}.json').read_text()) for s in SEEDS}
  rows[mode]['spherical_kmeans3']={(r,s):json.loads((a.root/f'results/{mode}/spherical_kmeans3_rseed{r}/ipc3_sseed{s}.json').read_text()) for r in (0,1) for s in SEEDS}
  base=a.r0_full if mode=='fullframe_cutmix' else a.r0_rrc
  r0[mode]={s:json.loads((base/f'ipc3_sseed{s}.json').read_text()) for s in SEEDS}
  for source in rows[mode].values():
   for row in source.values():
    assert row['epochs']==400 and row['temperature']==20 and row['mix_type']=='cutmix' and not row['fkd_hard_label']
 groups={}
 for mode in MODES:
  groups[mode]={
   'r0':{m:stats([r0[mode][s][m] for s in SEEDS]) for m in METRICS},
   'global_center_top3':{m:stats([rows[mode]['global_center_top3'][s][m] for s in SEEDS]) for m in METRICS},
   'spherical_kmeans3':{m:stats([rows[mode]['spherical_kmeans3'][r,s][m] for r in (0,1) for s in SEEDS]) for m in METRICS}}
 contrasts={}
 for method in ('global_center_top3','spherical_kmeans3'):
  contrasts[f'{method}: rrc minus fullframe']={}
  for m in METRICS:
   if method=='global_center_top3':v=[rows['rrc_cutmix'][method][s][m]-rows['fullframe_cutmix'][method][s][m] for s in SEEDS]
   else:v=[rows['rrc_cutmix'][method][r,s][m]-rows['fullframe_cutmix'][method][r,s][m] for r in (0,1) for s in SEEDS]
   contrasts[f'{method}: rrc minus fullframe'][m]=stats(v)
 for mode in MODES:
  top=rows[mode]['global_center_top3'];km=rows[mode]['spherical_kmeans3']
  kmmean={s:{m:statistics.mean(km[r,s][m] for r in (0,1)) for m in METRICS} for s in SEEDS}
  for label,values in (
   ('global_center_top3 minus r0',{m:[top[s][m]-r0[mode][s][m] for s in SEEDS] for m in METRICS}),
   ('mean_kmeans3 minus r0',{m:[kmmean[s][m]-r0[mode][s][m] for s in SEEDS] for m in METRICS}),
   ('mean_kmeans3 minus global_center_top3',{m:[kmmean[s][m]-top[s][m] for s in SEEDS] for m in METRICS})):
   contrasts[f'{mode}: {label}']={m:stats(v) for m,v in values.items()}
 result={'status':'complete','protocol':'dino_ipc3_soft_v2_fullframe_rrc_factorial','groups':groups,'paired_contrasts':contrasts,
         'teacher_seed':42,'student_seeds':list(SEEDS),'kmeans_selection_seeds':[0,1],'new_fkd_sets':6,'new_student_runs':18}
 out=a.root/'summary/summary.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
