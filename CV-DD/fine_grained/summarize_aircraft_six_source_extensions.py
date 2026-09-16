"""Summarize uniform-compression control and BBox source-seed extension."""
import argparse,json,statistics
from pathlib import Path
SEEDS=(42,43,44)
def stats(v):v=list(map(float,v));return {'mean':statistics.mean(v),'sample_sd':statistics.stdev(v),'values':v}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--parent',required=True,type=Path);a=p.parse_args();errors=[]
 uniform_soft={};uniform_hard={};bbox={0:{}}
 for s in SEEDS:
  try:
   uniform_soft[s]=json.loads((a.root/f'results/soft_uniform/sseed{s}.json').read_text());uniform_hard[s]=json.loads((a.root/f'results/hard_uniform/sseed{s}.json').read_text())
   bbox[0][s]=json.loads((a.parent/f'results/hard/compressed/sseed{s}.json').read_text())
  except Exception as e:errors.append({'seed':s,'error':repr(e)})
 for r in (1,2):
  bbox[r]={}
  for s in SEEDS:
   try:bbox[r][s]=json.loads((a.root/f'results/hard_bbox_rseed{r}/sseed{s}.json').read_text())
   except Exception as e:errors.append({'rseed':r,'seed':s,'error':repr(e)})
 ref_soft={s:json.loads((a.parent/f'results/soft/reference/sseed{s}.json').read_text()) for s in SEEDS};bbox_soft={s:json.loads((a.parent/f'results/soft/compressed/sseed{s}.json').read_text()) for s in SEEDS};ref_hard={s:json.loads((a.parent/f'results/hard/reference/sseed{s}.json').read_text()) for s in SEEDS};bbox_hard=bbox[0]
 groups={'soft':{},'hard':{}}
 for protocol,rows_ref,rows_bbox,rows_uni,bk,fk in [('soft',ref_soft,bbox_soft,uniform_soft,'best_top1','final_epoch_top1'),('hard',ref_hard,bbox_hard,uniform_hard,'best_top1_diagnostic','final_top1')]:
  for name,rows in [('reference',rows_ref),('uniform112',rows_uni),('bbox112',rows_bbox)]:groups[protocol][name]={'best':stats(rows[s][bk] for s in SEEDS),'final':stats(rows[s][fk] for s in SEEDS)}
  groups[protocol]['bbox_minus_uniform']={'best':stats(rows_bbox[s][bk]-rows_uni[s][bk] for s in SEEDS),'final':stats(rows_bbox[s][fk]-rows_uni[s][fk] for s in SEEDS)}
  groups[protocol]['uniform_minus_reference']={'best':stats(rows_uni[s][bk]-rows_ref[s][bk] for s in SEEDS),'final':stats(rows_uni[s][fk]-rows_ref[s][fk] for s in SEEDS)}
 by_source={str(r):{'best':stats(bbox[r][s]['best_top1_diagnostic'] for s in SEEDS),'final':stats(bbox[r][s]['final_top1'] for s in SEEDS)} for r in (0,1,2)}
 all_rows=[bbox[r][s] for r in (0,1,2) for s in SEEDS]
 extension={'by_source_seed':by_source,'all9':{'best':stats(x['best_top1_diagnostic'] for x in all_rows),'final':stats(x['final_top1'] for x in all_rows)},'claim_scope':'absolute BBox stability across source selections; rseed1/2 lack matched three-source Hard-v1 R0, so +9.83 gain is not paired outside rseed0'}
 result={'status':'complete' if not errors else 'failed','protocol':'aircraft_six_source_uniform_control_and_bbox_seed_extension_v1','groups':groups,'bbox_hard_source_extension':extension,'new_fkd_sets':1,'new_student_runs':12,'errors':errors}
 out=a.root/'summary/summary.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 if result['status']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
