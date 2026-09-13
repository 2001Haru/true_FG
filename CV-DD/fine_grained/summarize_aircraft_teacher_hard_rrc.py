import argparse,json,os,statistics
from pathlib import Path

def stats(values):
    values=list(map(float,values));return {"mean":statistics.mean(values),"sample_std":statistics.stdev(values),"values":values}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--soft-root',type=Path,required=True);p.add_argument('--hard-root',type=Path,required=True);a=p.parse_args()
    rows=[];errors=[]
    for method,source in (("random_real",0),("original_rded",42)):
        for student in (42,43,44):
            path=a.root/f"results/{method}/source_seed{source}/ipc3_sseed{student}.json"
            try:
                x=json.loads(path.read_text())
                assert x['training_target']=='fkd_teacher_argmax_ce'
                assert x['fkd_teacher_hard_label'] is True and x['mix_type'] is None
                assert x['student_protocol_name']=='standard_protocol_v2_teacher_hard'
                assert x['student_seed']==student and x['batch_size']==20 and x['gradient_accumulation_steps']==2
                assert x['epochs']==400 and x['scheduler_t_max']==400
                rows.append({"method":method,"source_seed":source,"student_seed":student,"best":x['best_top1'],"final":x['final_epoch_top1'],"best_epoch":x['best_epoch'],"initial_hash":x['initial_model_sha256']})
            except Exception as exc: errors.append({"path":str(path),"error":repr(exc)})
    groups=[]
    for method,source in (("random_real",0),("original_rded",42)):
        current=[r for r in rows if r['method']==method]
        soft=[];hard=[]
        for student in (42,43,44):
            soft.append(json.loads((a.soft_root/f"results/rrc_no_cutmix/{method}/source_seed{source}/ipc3_sseed{student}.json").read_text()))
            hard.append(json.loads((a.hard_root/f"results/rrc_no_cutmix/{method}/source_seed{source}/ipc3_sseed{student}.json").read_text()))
        groups.append({"method":method,"source_seed":source,"teacher_hard_best":stats(r['best'] for r in current),"teacher_hard_final":stats(r['final'] for r in current),"best_epochs":[r['best_epoch'] for r in current],"original_hard_best":stats(x['best_top1'] for x in hard),"original_hard_final":stats(x['final_epoch_top1'] for x in hard),"soft_best":stats(x['best_top1'] for x in soft),"soft_final":stats(x['final_epoch_top1'] for x in soft),"teacher_hard_minus_original_hard_best":stats(r['best']-h['best_top1'] for r,h in zip(current,hard)),"teacher_hard_minus_original_hard_final":stats(r['final']-h['final_epoch_top1'] for r,h in zip(current,hard))})
    for student in (42,43,44):
        hashes={r['initial_hash'] for r in rows if r['student_seed']==student}
        if len(hashes)!=1: errors.append({"student_seed":student,"error":"A/B initialization hash mismatch"})
    audits={}
    for method in ('random_real','original_rded'):
        try: audits[method]=json.loads((a.root/f"audits/{method}.json").read_text())
        except Exception as exc: errors.append({"audit":method,"error":repr(exc)})
    result={"status":"complete" if len(rows)==6 and not errors else "failed","dataset":"A_imsize224","ipc":3,"view":"RRC_[0.08,1]+flip_no_CutMix","target":"per-view cached Teacher raw-logit argmax; T1 CE","rows":rows,"groups":groups,"audits":audits,"errors":errors}
    out=a.root/'summary/aircraft_teacher_hard_rrc.json';tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');os.replace(tmp,out);print(json.dumps({"status":result['status'],"rows":len(rows),"errors":errors},indent=2));
    if result['status']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
