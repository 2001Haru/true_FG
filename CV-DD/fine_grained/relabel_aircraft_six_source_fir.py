"""Generate F1 FIR-preemphasized FKD by replaying the frozen BBox-6 trajectory."""
import argparse,json,os,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from paired_source_dataset import PairedEpochSampler,PairedSoftSaveDataset
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from relabel.utils_fkd import mix_aug
from fine_grained.rescore_fkd_views import load_teacher,sha256
def atomic(x,p):p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2)+'\n');os.replace(q,p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path);p.add_argument('--source-fkd',required=True,type=Path);p.add_argument('--output-fkd',required=True,type=Path);p.add_argument('--teacher',required=True,type=Path);p.add_argument('--epochs',type=int,default=400);p.add_argument('--workers',type=int,default=8);p.add_argument('--batch-size',type=int,default=20);p.add_argument('--classes',type=int,default=100);p.add_argument('--image-mode',choices=('reference','compressed','compressed_fir'),default='compressed_fir');a=p.parse_args()
 dataset=PairedSoftSaveDataset(a.manifest,schedule_seed=42,compressed_mode=a.image_mode);sampler=PairedEpochSampler(dataset,42);loader=DataLoader(dataset,batch_size=a.batch_size,sampler=sampler,num_workers=a.workers,persistent_workers=a.workers>0,pin_memory=True);source_count=int(dataset.reference.manifest.get('unique_sources',600))
 teacher=load_teacher(a.teacher,a.classes);has_source=(a.source_fkd.resolve()!=a.output_fkd.resolve() and (a.source_fkd/'relabel_manifest.json').is_file());mixargs=SimpleNamespace(mode='fkd_load' if has_source else 'fkd_save',mix_type='cutmix',cutmix=1.,mixup=.8)
 view_order=('select source -> recorded flip -> recorded CutMix' if a.image_mode=='reference' else
             'select strip -> inverse decode -> recorded flip -> recorded CutMix' if a.image_mode=='compressed' else
             'select strip -> inverse decode -> frozen axis FIR -> recorded flip -> recorded CutMix')
 first_split=a.batch_size//2
 m={'status':'running','protocol':'fg_paired_replay_soft_v1','epochs':a.epochs,'batch_size':a.batch_size,'parents':len(dataset),'sources':source_count,'storage_ipc':int(dataset.reference.manifest['storage_ipc']),'classes':a.classes,'paired_source_manifest':str(a.manifest.resolve()),'paired_source_manifest_sha256':sha256(a.manifest),'source_fkd':str(a.source_fkd),'image_mode':a.image_mode,'source_schedule':'recorded parent/subsource entries','view_order':view_order,'rrc':False,'teacher_path':str(a.teacher.resolve()),'teacher_sha256':sha256(a.teacher),'teacher_mode':'train','teacher_forward_split':[first_split,a.batch_size-first_split],'temperature':20};atomic(m,a.output_fkd/'relabel_manifest.json')
 for epoch in range(a.epochs):
  dataset.set_epoch(epoch);out=a.output_fkd/f'epoch_{epoch}';out.mkdir(parents=True,exist_ok=True)
  for batch,data in enumerate(loader):
   _,images,_,flip,coords,sources,parents=data;images=images.cuda(non_blocking=True)
   if has_source:
    c=torch.load(a.source_fkd/f'epoch_{epoch}/batch_{batch}.tar',map_location='cpu',weights_only=False);mix_index,mix_lam,mix_bbox=c[2],c[3],c[4]
    if not torch.equal(sources,c[6]) or not torch.equal(parents,c[7]) or not torch.equal(flip,c[1]):raise RuntimeError(f'metadata mismatch epoch{epoch} batch{batch}')
    mixed,_,_,_=mix_aug(images,mixargs,mix_index,mix_lam,mix_bbox)
   else:
    mixed,mix_index,mix_lam,mix_bbox=mix_aug(images,mixargs)
   split=max(1,mixed.shape[0]//2);chunks=[teacher(mixed[:split])]
   if split<mixed.shape[0]:chunks.append(teacher(mixed[split:]))
   logits=torch.cat(chunks,0).half().cpu();torch.save([coords,flip,mix_index,mix_lam,mix_bbox,logits,sources,parents],out/f'batch_{batch}.tar')
  print(f'epoch={epoch}',flush=True)
 files=list(a.output_fkd.glob('epoch_*/batch_*.tar'));expected=a.epochs*((len(dataset)+a.batch_size-1)//a.batch_size)
 if len(files)!=expected:raise RuntimeError((len(files),expected))
 m['status']='complete';m['batch_files']=len(files);atomic(m,a.output_fkd/'relabel_manifest.json');print(json.dumps({'status':'complete','files':len(files)}))
if __name__=='__main__':main()
