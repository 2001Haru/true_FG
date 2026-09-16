"""Summarize six-source paired reference versus BBox strip encoding."""

import argparse, json, statistics
from pathlib import Path

SEEDS=(42,43,44); MODES=("reference","compressed")
def stats(v): v=list(map(float,v)); return {"mean":statistics.mean(v),"sample_sd":statistics.stdev(v),"values":v}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();soft={};hard={};errors=[]
 for mode in MODES:
  for seed in SEEDS:
   try:
    soft[(mode,seed)]=json.loads((a.root/f'results/soft/{mode}/sseed{seed}.json').read_text())
    hard[(mode,seed)]=json.loads((a.root/f'results/hard/{mode}/sseed{seed}.json').read_text())
    assert soft[(mode,seed)]['paired_source_mode']==mode and soft[(mode,seed)]['student_seed']==seed
    assert hard[(mode,seed)]['paired_source_mode']==mode and hard[(mode,seed)]['student_seed']==seed
    assert hard[(mode,seed)]['updates_completed']==3000 and hard[(mode,seed)]['examples_seen']==180000
    assert hard[(mode,seed)]['paired_source_trajectory_audit']['source_exposure_min']==300
    assert hard[(mode,seed)]['paired_source_trajectory_audit']['source_exposure_max']==300
   except Exception as e: errors.append({'mode':mode,'seed':seed,'error':repr(e)})
 groups={}
 for protocol,rows,best_key,final_key in (("soft",soft,"best_top1","final_epoch_top1"),("hard",hard,"best_top1_diagnostic","final_top1")):
  groups[protocol]={}
  for mode in MODES:
   groups[protocol][mode]={'best':stats(rows[(mode,s)][best_key] for s in SEEDS),'final':stats(rows[(mode,s)][final_key] for s in SEEDS)}
  groups[protocol]['compressed_minus_reference']={'best':stats(rows[("compressed",s)][best_key]-rows[("reference",s)][best_key] for s in SEEDS),'final':stats(rows[("compressed",s)][final_key]-rows[("reference",s)][final_key] for s in SEEDS)}
 result={'status':'complete' if not errors else 'failed','protocol':'aircraft_six_source_bbox_pack_soft_hard_v1','groups':groups,
         'soft_schedule':json.loads((a.root/'audits/soft_fkd.json').read_text()),
         'hard_exposure':{'examples_per_run':180000,'sources':600,'exact_exposure_per_source':300,
                          'reason':'300 parents x 600 complete epochs; 5 batches/epoch with final batch44; 3000 updates'},
         'new_fkd_sets':2,'new_student_runs':12,'errors':errors}
 out=a.root/'summary/summary.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 if result['status']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
