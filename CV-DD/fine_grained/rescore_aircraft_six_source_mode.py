"""Rescore one six-source image mode using authoritative cached source and augmentation metadata."""
import argparse,json,os
from pathlib import Path
from types import SimpleNamespace
import torch
from paired_source_dataset import AIRCRAFT_MEAN,AIRCRAFT_STD,PairedSourceIndex
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from relabel.utils_fkd import mix_aug
from fine_grained.rescore_fkd_views import load_teacher,sha256
from torchvision.transforms import functional as TF
def atomic(x,p):p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.json.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True,type=Path);p.add_argument('--mode',required=True,choices=('uniform','compressed','reference'));p.add_argument('--source-fkd',required=True,type=Path);p.add_argument('--output-fkd',required=True,type=Path);p.add_argument('--teacher',required=True,type=Path);a=p.parse_args()
 index=PairedSourceIndex(a.manifest,a.mode);teacher=load_teacher(a.teacher,100);mix_args=SimpleNamespace(mode='fkd_load',mix_type='cutmix',cutmix=1.,mixup=.8)
 meta={'status':'running','protocol':'aircraft_six_source_rescore_v1','image_mode':a.mode,'epochs':400,'batch_size':20,'parents':300,'sources':600,'rrc':False,'paired_source_manifest':str(a.manifest.resolve()),'paired_source_manifest_sha256':sha256(a.manifest),'source_fkd':str(a.source_fkd.resolve()),'teacher_sha256':sha256(a.teacher),'metadata_replay':['parent','subsource','flip','CutMix'],'view_order':'cached subsource -> decode224 -> cached flip -> cached CutMix'};atomic(meta,a.output_fkd/'relabel_manifest.json')
 for epoch in range(400):
  out=a.output_fkd/f'epoch_{epoch}';out.mkdir(parents=True,exist_ok=True)
  for batch in range(15):
   config=torch.load(a.source_fkd/f'epoch_{epoch}/batch_{batch}.tar',map_location='cpu',weights_only=False);coords,flip,mix_index,mix_lam,mix_bbox,_,sources,parents=config
   images=[]
   for parent,source,do_flip in zip(parents.tolist(),sources.tolist(),flip.tolist()):
    image=index.load(parent,source)
    if do_flip:image=TF.hflip(image)
    images.append(TF.normalize(TF.pil_to_tensor(image).float().div_(255),AIRCRAFT_MEAN,AIRCRAFT_STD))
   images=torch.stack(images).cuda();mixed,_,_,_=mix_aug(images,mix_args,mix_index,mix_lam,mix_bbox);logits=torch.cat((teacher(mixed[:10]),teacher(mixed[10:])),0).half().cpu()
   torch.save([coords,flip,mix_index,mix_lam,mix_bbox,logits,sources,parents],out/f'batch_{batch}.tar')
  print(f'epoch={epoch}',flush=True)
 meta['status']='complete';meta['batch_files']=6000;atomic(meta,a.output_fkd/'relabel_manifest.json')
if __name__=='__main__':main()
