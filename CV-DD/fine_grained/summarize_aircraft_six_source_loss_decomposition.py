"""Summarize object/background loss decomposition and decode smoothing."""
import argparse,json,statistics
from pathlib import Path
S=(42,43,44);ARMS=('object_decoded','background_decoded','background_hsmooth')
def st(v):v=list(map(float,v));return {'mean':statistics.mean(v),'sample_sd':statistics.stdev(v),'values':v}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--parent',required=True,type=Path);a=p.parse_args();rows={};errors=[]
 for arm in ARMS:
  rows[arm]={}
  for s in S:
   try:rows[arm][s]=json.loads((a.root/f'results/{arm}/sseed{s}.json').read_text())
   except Exception as e:errors.append({'arm':arm,'seed':s,'error':repr(e)})
 ref={s:json.loads((a.parent/f'results/soft/reference/sseed{s}.json').read_text()) for s in S};bbox={s:json.loads((a.parent/f'results/soft/compressed/sseed{s}.json').read_text()) for s in S}
 groups={'reference':{'best':st(ref[s]['best_top1'] for s in S),'final':st(ref[s]['final_epoch_top1'] for s in S)},'bbox_full':{'best':st(bbox[s]['best_top1'] for s in S),'final':st(bbox[s]['final_epoch_top1'] for s in S)}}
 for arm in ARMS:groups[arm]={'best':st(rows[arm][s]['best_top1'] for s in S),'final':st(rows[arm][s]['final_epoch_top1'] for s in S),'minus_reference_final':st(rows[arm][s]['final_epoch_top1']-ref[s]['final_epoch_top1'] for s in S)}
 contrasts={'smooth_minus_bbox_final':st(rows['background_hsmooth'][s]['final_epoch_top1']-bbox[s]['final_epoch_top1'] for s in S),'bbox_minus_reference_final':st(bbox[s]['final_epoch_top1']-ref[s]['final_epoch_top1'] for s in S),'component_sum_minus_full_loss_final':st((rows['object_decoded'][s]['final_epoch_top1']-ref[s]['final_epoch_top1'])+(rows['background_decoded'][s]['final_epoch_top1']-ref[s]['final_epoch_top1'])-(bbox[s]['final_epoch_top1']-ref[s]['final_epoch_top1']) for s in S)}
 result={'status':'complete' if not errors else 'failed','protocol':'aircraft_six_source_bbox_loss_decomposition_v1','groups':groups,'contrasts':contrasts,'arm_semantics':{'object_decoded':'decoded object rows + original background','background_decoded':'original object rows + decoded background','background_hsmooth':'full decoded image; background rows horizontally box-smoothed with odd kernel≈1/density capped15; object and bottom8 rows unchanged'},'new_fkd_sets':3,'new_student_runs':9,'errors':errors}
 out=a.root/'summary/summary.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));
 if result['status']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
