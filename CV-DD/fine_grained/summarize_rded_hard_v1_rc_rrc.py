"""Summarize Aircraft IPC3 original-RDED under Hard-v1 mild RC and RRC."""
import argparse,json,statistics
from pathlib import Path

MODES=('mild_rc','rrc');GSEEDS=(42,43,44);SSEEDS=(42,43,44);METRICS=('best_top1_diagnostic','final_top1')
def stats(v):return {'mean':statistics.mean(v),'sample_sd':statistics.stdev(v),'values':v}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path);p.add_argument('--r0-root',required=True,type=Path);a=p.parse_args()
 rows={m:{(g,s):json.loads((a.root/f'results/{m}/gseed{g}/sseed{s}.json').read_text()) for g in GSEEDS for s in SSEEDS} for m in MODES}
 r0={m:[json.loads((a.r0_root/f'{m}/r0/sseed{s}.json').read_text()) for s in SSEEDS] for m in MODES}
 for mode in MODES:
  for row in rows[mode].values():
   assert row['protocol']==f'hard_label_v1_{mode}' and row['train_crop_mode']==mode and row['updates_completed']==3000
   assert not row['cutmix'] and not row['mixup']
 groups={m:{metric:stats([rows[m][k][metric] for k in sorted(rows[m])]) for metric in METRICS} for m in MODES}
 by_generation={str(g):{m:{metric:stats([rows[m][g,s][metric] for s in SSEEDS]) for metric in METRICS} for m in MODES} for g in GSEEDS}
 contrasts={'rrc minus mild_rc':{metric:stats([rows['rrc'][k][metric]-rows['mild_rc'][k][metric] for k in sorted(rows['rrc'])]) for metric in METRICS}}
 descriptive={m:{metric:groups[m][metric]['mean']-statistics.mean(r[metric] for r in r0[m]) for metric in METRICS} for m in MODES}
 result={'status':'complete','protocol':'original_rded_hard_v1_rc_rrc_aircraft_ipc3','groups':groups,'by_generation_seed':by_generation,
         'paired_contrasts':contrasts,'descriptive_rded_minus_r0_seed0':descriptive,'generation_seeds':list(GSEEDS),'student_seeds':list(SSEEDS),'new_student_runs':18}
 out=a.root/'summary/summary.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
