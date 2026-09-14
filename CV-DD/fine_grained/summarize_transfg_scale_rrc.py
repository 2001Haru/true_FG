"""Paired seed0 comparisons for the TransFG scale/RRC interventions."""
import json
import statistics
from pathlib import Path
import argparse


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a=p.parse_args()
    base=a.root.parent
    old=Path('/linxi/dataset/FG_SoftAugFactorial_v2/aircraft_ipc3_v1/results')
    roots={'entropy_match':a.root/'results/entropy_match', 'rrc_only':a.root/'results/rrc_only',
           'transfg_cutmix_T20':base/'results/transfg',
           'resnet_cutmix':old/'fullframe_cutmix/random_real/source_seed0',
           'resnet_rrc':old/'rrc_no_cutmix/random_real/source_seed0'}
    rows={k:[json.load(open(v/f'ipc3_sseed{s}.json')) for s in (42,43,44)] for k,v in roots.items()}
    metrics=('best_top1','final_epoch_top1')
    def stats(v): return {'mean':statistics.mean(v),'sample_sd':statistics.stdev(v),'values':v}
    out={'status':'complete','temperature':json.load(open(a.root/'audits/temperature.json')),
         'groups':{k:{m:stats([x[m] for x in v]) for m in metrics} for k,v in rows.items()},'paired_deltas':{}}
    for candidate,ref in [('entropy_match','transfg_cutmix_T20'),('entropy_match','resnet_cutmix'),('rrc_only','resnet_rrc'),('rrc_only','resnet_cutmix')]:
        out['paired_deltas'][candidate+' minus '+ref]={m:stats([x[m]-y[m] for x,y in zip(rows[candidate],rows[ref])]) for m in metrics}
    for arm in ('entropy_match','rrc_only'):
        for x,ref in zip(rows[arm],rows['transfg_cutmix_T20']):
            assert x['initial_model_sha256']==ref['initial_model_sha256']
            assert x['temperature']==20 and x['epochs']==400
    (a.root/'paired_summary.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__': main()
