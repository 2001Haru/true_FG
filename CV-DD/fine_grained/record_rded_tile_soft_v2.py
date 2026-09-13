"""Audit Aircraft RDED-tile standard-v2 Soft result."""
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
 p=argparse.ArgumentParser();p.add_argument('--result',required=True,type=Path);p.add_argument('--control-result',required=True,type=Path);p.add_argument('--fkd',required=True,type=Path);p.add_argument('--teacher',required=True,type=Path);p.add_argument('--quadrant-audit',required=True,type=Path);p.add_argument('--source-audit',required=True,type=Path);p.add_argument('--generation-seed',required=True,type=int);p.add_argument('--student-seed',required=True,type=int);a=p.parse_args()
 r,c=load(a.result),load(a.control_result);audit_payload(r,100,3333);expect(r['student_protocol_name'],'standard_protocol_v2','protocol');expect(r['student_initialization'],'imagenet-v1','init');expect(r['student_seed'],a.student_seed,'seed');expect(r['synthetic_data_path'],c['synthetic_data_path'],'parent mosaics');expect(r['batch_size'],20,'batch');expect(r['gradient_accumulation_steps'],2,'accum');expect(r['epochs'],400,'epochs');close(r['backbone_learning_rate'],1e-4,'bb lr');close(r['head_learning_rate'],1e-3,'head lr');close(r['weight_decay'],1e-5,'wd');close(r['temperature'],20,'T');expect(r['mix_type'],'cutmix','mix');expect(r['initial_model_sha256'],c['initial_model_sha256'],'paired init')
 m=load(a.fkd/'relabel_manifest.json');expect(m['status'],'complete','Relabel');expect(m['single_quadrant'],True,'tile');expect(m['full_image_resize'],True,'resize');expect(m['mix_type'],'cutmix','CutMix');expect(m['quadrant_schedule'],'stable random permutation per parent and four-epoch block','schedule');expect(m['teacher_mode'],'train','BSSL');expect(m['teacher_sha256'],sha(a.teacher),'teacher')
 q=load(a.quadrant_audit);expect(q['status'],'complete','tile audit');expect(q['schedule'],'stable random permutation per parent path and four-epoch block','tile schedule');expect(q['per_image_quadrant_counts_unique'],[[100,100,100,100]],'tile balance');expect(q['saved_schedule_mismatches'],0,'schedule mismatch')
 s=load(a.source_audit);expect(s['status'],'complete','source audit');expect(s['tile_positions_per_class'],12,'tile positions');expect(s['new_high_resolution_information'],0,'new information')
 r['rded_tile_soft_v2']={'status':'complete','generation_seed':a.generation_seed,'student_seed':a.student_seed,'tile_seed':42,'parent_sampler_images':300,'tile_positions_per_class':12,'one_tile_per_parent_visit':True,'control_result':str(a.control_result.resolve()),'control_result_sha256':sha(a.control_result),'quadrant_audit':str(a.quadrant_audit.resolve()),'source_audit':str(a.source_audit.resolve())}
 t=a.result.with_suffix('.json.tmp');t.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');os.replace(t,a.result)
if __name__=='__main__':main()
