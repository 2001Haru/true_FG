import argparse,hashlib,json,math,os
from pathlib import Path
from audit_result import audit_payload

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1048576),b''):digest.update(block)
    return digest.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--result',required=True,type=Path);p.add_argument('--manifest',required=True,type=Path);p.add_argument('--method',required=True);p.add_argument('--fkd',required=True,type=Path);p.add_argument('--student-seed',required=True,type=int);a=p.parse_args()
    x=json.loads(a.result.read_text());m=json.loads(a.manifest.read_text());audit_payload(x,100,3333,expected_training_target='fkd_soft_label')
    assert m['status']=='complete' and m['ipc']==3 and x['student_seed']==a.student_seed
    assert x['student_protocol_name']=='standard_protocol_v2_fgdd_v1' and x['student_initialization']=='imagenet-v1'
    assert x['mix_type']=='cutmix' and x['batch_size']==20 and x['gradient_accumulation_steps']==2 and x['epochs']==400
    assert math.isclose(x['backbone_learning_rate'],1e-4) and math.isclose(x['head_learning_rate'],1e-3)
    assert Path(x['fkd_path']).resolve()==a.fkd.resolve()
    relabel=json.loads((a.fkd/'relabel_manifest.json').read_text());assert relabel['status']=='complete' and relabel['teacher_mode']=='train' and relabel['mix_type']=='cutmix' and relabel['full_image_resize'] is True
    x['fgdd_v1']={'status':'complete','method':a.method,'selection_manifest':str(a.manifest.resolve()),'selection_manifest_sha256':sha(a.manifest),'student_seed':a.student_seed,'construction_teacher_queries_excluded_from_final_fkd_budget':True}
    tmp=a.result.with_suffix('.json.tmp');tmp.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(tmp,a.result)
if __name__=='__main__':main()
