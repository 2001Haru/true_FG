"""Audit one Aircraft IPC3 hard-label RRC/CutMix factorial result."""
import argparse,hashlib,json,math,os
from pathlib import Path
from audit_result import audit_payload
def load(p):return json.loads(Path(p).read_text())
def sha(p):
 d=hashlib.sha256()
 with Path(p).open('rb') as h:
  for c in iter(lambda:h.read(1048576),b''):d.update(c)
 return d.hexdigest()
def expect(a,b,n):
 if a!=b:raise RuntimeError(f'{n}: {a!r} != {b!r}')
def close(a,b,n):
 if not math.isclose(float(a),b,rel_tol=0,abs_tol=1e-12):raise RuntimeError(f'{n}: {a!r} != {b!r}')
def main():
 p=argparse.ArgumentParser();p.add_argument('--result',required=True,type=Path);p.add_argument('--source-image-result',required=True,type=Path)
 p.add_argument('--fkd',required=True,type=Path);p.add_argument('--teacher',required=True,type=Path);p.add_argument('--method',required=True,choices=('random_real','original_rded'))
 p.add_argument('--source-seed',required=True,type=int);p.add_argument('--student-seed',required=True,type=int);p.add_argument('--arm',required=True,choices=('rrc_no_cutmix','fullframe_cutmix'));a=p.parse_args()
 r,s=load(a.result),load(a.source_image_result);audit_payload(r,100,3333,expected_training_target='fkd_replay_hard_cutmix_ce')
 expect(r['student_protocol_name'],'standard_protocol_v2_hard_replay','protocol');expect(r['student_initialization'],'imagenet-v1','init');expect(r['student_seed'],a.student_seed,'seed')
 expect(r['synthetic_data_path'],s['synthetic_data_path'],'images');expect(r['fkd_path'],str(a.fkd.resolve()),'FKD');expect(r['batch_size'],20,'batch');expect(r['gradient_accumulation_steps'],2,'accum')
 expect(r['epochs'],400,'epochs');close(r['backbone_learning_rate'],1e-4,'bb lr');close(r['head_learning_rate'],1e-3,'head lr');close(r['weight_decay'],1e-5,'wd');expect(r['scheduler_t_max'],400,'Tmax')
 m=load(a.fkd/'relabel_manifest.json');expect(m['status'],'complete','Relabel');expect(m['teacher_mode'],'train','BSSL');expect(m['teacher_sha256'],sha(a.teacher),'teacher')
 if a.arm=='rrc_no_cutmix':
  expect(m['full_image_resize'],False,'fullframe');expect(m['mix_type'],None,'mix');close(m['min_scale_crops'],.08,'minscale');expect(r['mix_type'],None,'result mix');expect(r['hard_cutmix_lambda_source'],None,'lambda')
 else:
  expect(m['full_image_resize'],True,'fullframe');expect(m['mix_type'],'cutmix','mix');expect(r['mix_type'],'cutmix','result mix');expect(r['hard_cutmix_lambda_source'],'actual_bbox_area','lambda')
 f=load(a.fkd/'fkd_audit.json');expect(f['status'],'complete','FKD audit');expect(f['images'],300,'images');expect(f['epochs'],400,'FKD epochs')
 r['hard_aug_factorial']={'status':'complete','arm':a.arm,'method':a.method,'source_seed':a.source_seed,'student_seed':a.student_seed,'source_image_result':str(a.source_image_result.resolve()),'source_image_result_sha256':sha(a.source_image_result)}
 t=a.result.with_suffix('.json.tmp');t.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');os.replace(t,a.result)
if __name__=='__main__':main()
